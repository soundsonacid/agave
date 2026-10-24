//! `sol_math`: run one short stack program per lane over up to three inputs.
//!
//! Proof of concept for a vectorizable arithmetic syscall. The op sequence is a `u64`:
//! bits 0..3 are flags, bits 4..63 hold up to ten 6-bit tokens, executed in order until
//! the first `END` token.
//!
//! ```text
//! bit 0      lane width: 0 = u64, 1 = u128 (inputs and output alike)
//! bits 1..3  inputs[0..3]: 0 = one value, broadcast to every lane; 1 = `lane_count` lanes
//! ```
//!
//! A token's class (bits 5..4) fixes its stack effect; the index (bits 3..0) picks the op:
//!
//! ```text
//! push    (+1)  END=0  A=1 B=2 C=3  LIT=4 (next token is the value, 0..63)  DUP=5
//! unary   ( 0)  NOT=16 NEG=17
//! binary  (-1)  ADD=32 SUB AND OR XOR MIN MAX LSH LRSH ARSH DIV MOD MUL_LO52 MUL_HI52=45
//! ternary (-2)  MUL128_DIV=48 MUL128_SHR=49 SELECT=50
//! ```
//!
//! Semantics: lanes are unsigned containers. ADD, SUB and NEG wrap. MIN, MAX and ARSH are
//! signed; everything else is unsigned. DIV and MOD fail the syscall on a zero divisor. A
//! shift count at or past the lane width gives 0 (ARSH: the sign fill). MUL_LO52 and
//! MUL_HI52 multiply the low 52 bits of each operand and return the low and high 52 bits of
//! the product. MUL128_SHR and MUL128_DIV take the full double-width product, then shift or
//! divide, and truncate the result to the lane width. Binary ops compute `x op y` with `y`
//! on top of the stack; SELECT is `cond then else SELECT` and tests `cond != 0`.
//!
//! Cost: `MATH_BASE_COST + ceil(lane_count / 8) * words_per_lane * tokens`.
//!
//! The output may be the same buffer as a vector input (an in-place update) or disjoint from
//! every input; any other overlap fails the syscall, so the scalar and AVX-512 paths agree.
//!
//! Host execution: on x86-64 with AVX-512F + IFMA, u64 lanes run eight at a time with the
//! stack held in `zmm` registers (`avx512` module); everywhere else, and for u128 lanes, the
//! same program runs one lane at a time.

use {super::*, std::ptr};

const FLAG_BITS: u32 = 4;
const FLAG_WIDE: u64 = 1 << 0;
const FLAG_VECTOR_A: u64 = 1 << 1;
const FLAG_VECTOR_B: u64 = 1 << 2;
const FLAG_VECTOR_C: u64 = 1 << 3;

const TOKEN_BITS: u32 = 6;
const TOKEN_MASK: u64 = (1 << TOKEN_BITS) - 1;
/// Tokens that fit in the op word after the flag nibble.
pub const MAX_TOKENS: usize = 10;

pub const END: u8 = 0;
pub const A: u8 = 1;
pub const B: u8 = 2;
pub const C: u8 = 3;
pub const LIT: u8 = 4;
pub const DUP: u8 = 5;
pub const NOT: u8 = 16;
pub const NEG: u8 = 17;
pub const ADD: u8 = 32;
pub const SUB: u8 = 33;
pub const AND: u8 = 34;
pub const OR: u8 = 35;
pub const XOR: u8 = 36;
pub const MIN: u8 = 37;
pub const MAX: u8 = 38;
pub const LSH: u8 = 39;
pub const LRSH: u8 = 40;
pub const ARSH: u8 = 41;
pub const DIV: u8 = 42;
pub const MOD: u8 = 43;
pub const MUL_LO52: u8 = 44;
pub const MUL_HI52: u8 = 45;
pub const MUL128_DIV: u8 = 48;
pub const MUL128_SHR: u8 = 49;
pub const SELECT: u8 = 50;

/// Charged once per call, before any lane work.
pub const MATH_BASE_COST: u64 = 100;
/// Lanes charged as one unit: one 512-bit vector of u64.
pub const MATH_LANES_PER_UNIT: u64 = 8;

const MASK_52: u128 = (1 << 52) - 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Token {
    PushA,
    PushB,
    PushC,
    Lit(u8),
    Dup,
    Not,
    Neg,
    Add,
    Sub,
    And,
    Or,
    Xor,
    Min,
    Max,
    Lsh,
    Lrsh,
    Arsh,
    Div,
    Mod,
    MulLo52,
    MulHi52,
    Mul128Div,
    Mul128Shr,
    Select,
}

impl Token {
    /// Stack values consumed; every token pushes exactly one.
    fn operands(self) -> usize {
        match self {
            Token::PushA | Token::PushB | Token::PushC | Token::Lit(_) => 0,
            Token::Dup | Token::Not | Token::Neg => 1,
            Token::Add
            | Token::Sub
            | Token::And
            | Token::Or
            | Token::Xor
            | Token::Min
            | Token::Max
            | Token::Lsh
            | Token::Lrsh
            | Token::Arsh
            | Token::Div
            | Token::Mod
            | Token::MulLo52
            | Token::MulHi52 => 2,
            Token::Mul128Div | Token::Mul128Shr | Token::Select => 3,
        }
    }
}

/// A decoded, validated op word.
#[derive(Clone, Copy, Debug)]
pub struct OpSequence {
    tokens: [Token; MAX_TOKENS],
    len: usize,
    wide: bool,
    vector_inputs: [bool; 3],
}

impl OpSequence {
    pub fn decode(word: u64) -> Result<Self, Error> {
        let invalid = || Error::from(SyscallError::InvalidMathSequence);
        let mut tokens = [Token::Dup; MAX_TOKENS];
        let mut len = 0;
        let mut depth = 0usize;
        let mut remaining = word >> FLAG_BITS;
        let mut position = 0;
        while position < MAX_TOKENS {
            let value = (remaining & TOKEN_MASK) as u8;
            remaining >>= TOKEN_BITS;
            position += 1;
            let token = match value {
                END => break,
                A => Token::PushA,
                B => Token::PushB,
                C => Token::PushC,
                LIT => {
                    if position >= MAX_TOKENS {
                        return Err(invalid());
                    }
                    let literal = (remaining & TOKEN_MASK) as u8;
                    remaining >>= TOKEN_BITS;
                    position += 1;
                    Token::Lit(literal)
                }
                DUP => Token::Dup,
                NOT => Token::Not,
                NEG => Token::Neg,
                ADD => Token::Add,
                SUB => Token::Sub,
                AND => Token::And,
                OR => Token::Or,
                XOR => Token::Xor,
                MIN => Token::Min,
                MAX => Token::Max,
                LSH => Token::Lsh,
                LRSH => Token::Lrsh,
                ARSH => Token::Arsh,
                DIV => Token::Div,
                MOD => Token::Mod,
                MUL_LO52 => Token::MulLo52,
                MUL_HI52 => Token::MulHi52,
                MUL128_DIV => Token::Mul128Div,
                MUL128_SHR => Token::Mul128Shr,
                SELECT => Token::Select,
                _ => return Err(invalid()),
            };
            if depth < token.operands() {
                return Err(invalid());
            }
            depth = depth - token.operands() + 1;
            tokens[len] = token;
            len += 1;
        }
        if len == 0 || depth != 1 {
            return Err(invalid());
        }
        Ok(Self {
            tokens,
            len,
            wide: word & FLAG_WIDE != 0,
            vector_inputs: [
                word & FLAG_VECTOR_A != 0,
                word & FLAG_VECTOR_B != 0,
                word & FLAG_VECTOR_C != 0,
            ],
        })
    }

    /// Tokens executed per lane (a `LIT` and its value count as one).
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn evaluate<L: Lane>(&self, a: L, b: L, c: L) -> Result<L, Error> {
        let mut stack = [L::ZERO; MAX_TOKENS];
        let mut depth = 0usize;
        for token in &self.tokens[..self.len] {
            let operands = token.operands();
            let base = depth - operands;
            let result = match *token {
                Token::PushA => a,
                Token::PushB => b,
                Token::PushC => c,
                Token::Lit(value) => L::from_small(value),
                Token::Dup => stack[base],
                Token::Not => !stack[base],
                Token::Neg => stack[base].wrapping_neg(),
                Token::Add => stack[base].wrapping_add(stack[base + 1]),
                Token::Sub => stack[base].wrapping_sub(stack[base + 1]),
                Token::And => stack[base] & stack[base + 1],
                Token::Or => stack[base] | stack[base + 1],
                Token::Xor => stack[base] ^ stack[base + 1],
                Token::Min => stack[base].min_signed(stack[base + 1]),
                Token::Max => stack[base].max_signed(stack[base + 1]),
                Token::Lsh => stack[base].shift_left(stack[base + 1]),
                Token::Lrsh => stack[base].shift_right_logical(stack[base + 1]),
                Token::Arsh => stack[base].shift_right_arithmetic(stack[base + 1]),
                Token::Div => stack[base].checked_div(stack[base + 1]).ok_or(SyscallError::MathDivideByZero)?,
                Token::Mod => stack[base].checked_rem(stack[base + 1]).ok_or(SyscallError::MathDivideByZero)?,
                Token::MulLo52 => stack[base].mul_lo52(stack[base + 1]),
                Token::MulHi52 => stack[base].mul_hi52(stack[base + 1]),
                Token::Mul128Div => stack[base]
                    .mul_wide_div(stack[base + 1], stack[base + 2])
                    .ok_or(SyscallError::MathDivideByZero)?,
                Token::Mul128Shr => stack[base].mul_wide_shr(stack[base + 1], stack[base + 2]),
                Token::Select => {
                    if stack[base].is_zero() {
                        stack[base + 2]
                    } else {
                        stack[base + 1]
                    }
                }
            };
            stack[base] = result;
            depth = base + 1;
        }
        Ok(stack[0])
    }
}

/// Build an op word: `flags` is the low nibble, `tokens` are raw token values in order
/// (a `LIT` value is its own entry). Does not validate; `OpSequence::decode` does.
pub fn encode(flags: u8, tokens: &[u8]) -> u64 {
    let mut word = flags as u64 & ((1 << FLAG_BITS) - 1);
    for (position, token) in tokens.iter().enumerate() {
        word |= (*token as u64 & TOKEN_MASK) << (FLAG_BITS + TOKEN_BITS * position as u32);
    }
    word
}

/// Which host implementation u64 lanes run on here: `"avx512"` or `"scalar"`.
pub fn host_vector_path() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    if avx512::available() {
        return "avx512";
    }
    "scalar"
}

/// Host-side reference: evaluate one u64 lane.
pub fn evaluate_u64(word: u64, a: u64, b: u64, c: u64) -> Result<u64, Error> {
    OpSequence::decode(word)?.evaluate(a, b, c)
}

/// Host-side reference: evaluate one u128 lane.
pub fn evaluate_u128(word: u64, a: u128, b: u128, c: u128) -> Result<u128, Error> {
    OpSequence::decode(word)?.evaluate(a, b, c)
}

/// The arithmetic each op needs, for the two lane widths.
trait Lane:
    Copy
    + std::ops::Not<Output = Self>
    + std::ops::BitAnd<Output = Self>
    + std::ops::BitOr<Output = Self>
    + std::ops::BitXor<Output = Self>
{
    const ZERO: Self;
    /// u64 words per lane in VM memory.
    const WORDS: usize;

    fn from_small(value: u8) -> Self;
    fn is_zero(self) -> bool;
    fn wrapping_neg(self) -> Self;
    fn wrapping_add(self, other: Self) -> Self;
    fn wrapping_sub(self, other: Self) -> Self;
    fn min_signed(self, other: Self) -> Self;
    fn max_signed(self, other: Self) -> Self;
    fn shift_left(self, count: Self) -> Self;
    fn shift_right_logical(self, count: Self) -> Self;
    fn shift_right_arithmetic(self, count: Self) -> Self;
    fn checked_div(self, divisor: Self) -> Option<Self>;
    fn checked_rem(self, divisor: Self) -> Option<Self>;
    fn mul_lo52(self, other: Self) -> Self;
    fn mul_hi52(self, other: Self) -> Self;
    fn mul_wide_shr(self, other: Self, shift: Self) -> Self;
    fn mul_wide_div(self, other: Self, divisor: Self) -> Option<Self>;

    /// # Safety
    /// `source` must point at `Self::WORDS` readable u64s.
    unsafe fn read(source: *const u64) -> Self;
    /// # Safety
    /// `destination` must point at `Self::WORDS` writable u64s.
    unsafe fn write(self, destination: *mut u64);
}

/// Shift count as a usable `u32`, or `None` when it is at or past the lane width.
fn shift_count(count: u128, bits: u32) -> Option<u32> {
    (count < bits as u128).then_some(count as u32)
}

impl Lane for u64 {
    const ZERO: Self = 0;
    const WORDS: usize = 1;

    fn from_small(value: u8) -> Self {
        value as u64
    }
    fn is_zero(self) -> bool {
        self == 0
    }
    fn wrapping_neg(self) -> Self {
        u64::wrapping_neg(self)
    }
    fn wrapping_add(self, other: Self) -> Self {
        u64::wrapping_add(self, other)
    }
    fn wrapping_sub(self, other: Self) -> Self {
        u64::wrapping_sub(self, other)
    }
    fn min_signed(self, other: Self) -> Self {
        (self as i64).min(other as i64) as u64
    }
    fn max_signed(self, other: Self) -> Self {
        (self as i64).max(other as i64) as u64
    }
    fn shift_left(self, count: Self) -> Self {
        shift_count(count as u128, Self::BITS).map_or(0, |count| self << count)
    }
    fn shift_right_logical(self, count: Self) -> Self {
        shift_count(count as u128, Self::BITS).map_or(0, |count| self >> count)
    }
    fn shift_right_arithmetic(self, count: Self) -> Self {
        let count = shift_count(count as u128, Self::BITS).unwrap_or(Self::BITS - 1);
        ((self as i64) >> count) as u64
    }
    fn checked_div(self, divisor: Self) -> Option<Self> {
        u64::checked_div(self, divisor)
    }
    fn checked_rem(self, divisor: Self) -> Option<Self> {
        u64::checked_rem(self, divisor)
    }
    fn mul_lo52(self, other: Self) -> Self {
        ((self as u128 & MASK_52) * (other as u128 & MASK_52) & MASK_52) as u64
    }
    fn mul_hi52(self, other: Self) -> Self {
        ((self as u128 & MASK_52) * (other as u128 & MASK_52) >> 52) as u64
    }
    fn mul_wide_shr(self, other: Self, shift: Self) -> Self {
        let product = self as u128 * other as u128;
        shift_count(shift as u128, 128).map_or(0, |shift| (product >> shift) as u64)
    }
    fn mul_wide_div(self, other: Self, divisor: Self) -> Option<Self> {
        (self as u128 * other as u128)
            .checked_div(divisor as u128)
            .map(|quotient| quotient as u64)
    }

    unsafe fn read(source: *const u64) -> Self {
        unsafe { ptr::read_unaligned(source) }
    }
    unsafe fn write(self, destination: *mut u64) {
        unsafe { ptr::write_unaligned(destination, self) }
    }
}

impl Lane for u128 {
    const ZERO: Self = 0;
    const WORDS: usize = 2;

    fn from_small(value: u8) -> Self {
        value as u128
    }
    fn is_zero(self) -> bool {
        self == 0
    }
    fn wrapping_neg(self) -> Self {
        u128::wrapping_neg(self)
    }
    fn wrapping_add(self, other: Self) -> Self {
        u128::wrapping_add(self, other)
    }
    fn wrapping_sub(self, other: Self) -> Self {
        u128::wrapping_sub(self, other)
    }
    fn min_signed(self, other: Self) -> Self {
        (self as i128).min(other as i128) as u128
    }
    fn max_signed(self, other: Self) -> Self {
        (self as i128).max(other as i128) as u128
    }
    fn shift_left(self, count: Self) -> Self {
        shift_count(count, Self::BITS).map_or(0, |count| self << count)
    }
    fn shift_right_logical(self, count: Self) -> Self {
        shift_count(count, Self::BITS).map_or(0, |count| self >> count)
    }
    fn shift_right_arithmetic(self, count: Self) -> Self {
        let count = shift_count(count, Self::BITS).unwrap_or(Self::BITS - 1);
        ((self as i128) >> count) as u128
    }
    fn checked_div(self, divisor: Self) -> Option<Self> {
        u128::checked_div(self, divisor)
    }
    fn checked_rem(self, divisor: Self) -> Option<Self> {
        u128::checked_rem(self, divisor)
    }
    fn mul_lo52(self, other: Self) -> Self {
        (self & MASK_52) * (other & MASK_52) & MASK_52
    }
    fn mul_hi52(self, other: Self) -> Self {
        (self & MASK_52) * (other & MASK_52) >> 52
    }
    fn mul_wide_shr(self, other: Self, shift: Self) -> Self {
        let (high, low) = mul_wide_u128(self, other);
        match shift_count(shift, 256) {
            None => 0,
            Some(0) => low,
            Some(shift) if shift >= 128 => high >> (shift - 128),
            Some(shift) => (low >> shift) | (high << (128 - shift)),
        }
    }
    fn mul_wide_div(self, other: Self, divisor: Self) -> Option<Self> {
        if divisor == 0 {
            return None;
        }
        let (high, low) = mul_wide_u128(self, other);
        Some(div_wide_u128(high, low, divisor))
    }

    unsafe fn read(source: *const u64) -> Self {
        unsafe {
            let low = ptr::read_unaligned(source);
            let high = ptr::read_unaligned(source.add(1));
            (high as u128) << 64 | low as u128
        }
    }
    unsafe fn write(self, destination: *mut u64) {
        unsafe {
            ptr::write_unaligned(destination, self as u64);
            ptr::write_unaligned(destination.add(1), (self >> 64) as u64);
        }
    }
}

/// Full 256-bit product as `(high, low)`.
fn mul_wide_u128(x: u128, y: u128) -> (u128, u128) {
    let (x_low, x_high) = (x as u64 as u128, x >> 64);
    let (y_low, y_high) = (y as u64 as u128, y >> 64);
    let low_low = x_low * y_low;
    let low_high = x_low * y_high;
    let high_low = x_high * y_low;
    let high_high = x_high * y_high;
    let (middle, middle_carry) = low_high.overflowing_add(high_low);
    let (low, low_carry) = low_low.overflowing_add(middle << 64);
    let high = high_high + (middle >> 64) + ((middle_carry as u128) << 64) + low_carry as u128;
    (high, low)
}

/// Low 128 bits of `(high:low) / divisor`, by shift-subtract long division.
fn div_wide_u128(high: u128, low: u128, divisor: u128) -> u128 {
    let mut remainder = 0u128;
    let mut quotient = 0u128;
    for bit in (0..256u32).rev() {
        let next_bit = if bit >= 128 {
            (high >> (bit - 128)) & 1
        } else {
            (low >> bit) & 1
        };
        // The remainder is below the divisor, so shifting it left loses at most one bit,
        // and that bit means the shifted remainder is above the divisor.
        let overflowed = remainder >> 127 != 0;
        remainder = (remainder << 1) | next_bit;
        if overflowed || remainder >= divisor {
            remainder = remainder.wrapping_sub(divisor);
            quotient = (quotient << 1) | 1;
        } else {
            quotient <<= 1;
        }
    }
    quotient
}

/// sol_math(op_sequence, lane_count, inputs: *const [*const u8; 3], output: *mut u8)
pub struct SyscallMath {}
impl BuiltinFunctionDefinition<InvokeContext<'_, '_>> for SyscallMath {
    type Error = Error;
    fn rust(
        invoke_context: &mut InvokeContext<'_, '_>,
        op_sequence: u64,
        lane_count: u64,
        inputs_addr: u64,
        output_addr: u64,
        _arg5: u64,
    ) -> Result<u64, Error> {
        let sequence = OpSequence::decode(op_sequence)?;
        let words_per_lane = if sequence.wide { u128::WORDS } else { u64::WORDS } as u64;
        let lane_units = lane_count
            .div_ceil(MATH_LANES_PER_UNIT)
            .saturating_mul(words_per_lane);
        invoke_context.consume_checked(
            MATH_BASE_COST.saturating_add(lane_units.saturating_mul(sequence.len() as u64)),
        )?;
        if lane_count == 0 {
            return Ok(0);
        }

        let check_aligned = invoke_context.get_check_aligned();
        let memory_mapping = invoke_context.memory_contexts.memory_mapping_mut()?;
        let input_addrs: [u64; 3] = translate_slice::<u64>(memory_mapping, inputs_addr, 3, check_aligned)?
            .try_into()
            .map_err(|_| SyscallError::InvalidLength)?;

        // The output may be an input (`acc += ...`), so the lane loops work through raw
        // pointers and read every input of a lane before writing that lane, as memmove does.
        let output_words = lane_count.saturating_mul(words_per_lane);
        touch_slice_mut::<MaybeUninit<u64>>(memory_mapping, output_addr, output_words)?;
        let output = translate_slice_inner!(
            memory_mapping,
            AccessType::Store,
            output_addr,
            output_words,
            MaybeUninit<u64>,
            check_aligned,
        )? as *mut MaybeUninit<u64> as *mut u64;
        let output_bytes = output_words as usize * size_of::<u64>();
        let mut inputs = [ptr::null::<u64>(); 3];
        let mut strides = [0usize; 3];
        for (index, input_addr) in input_addrs.into_iter().enumerate() {
            let is_vector = sequence.vector_inputs[index];
            let words = if is_vector { output_words } else { words_per_lane };
            let input = translate_slice::<u64>(memory_mapping, input_addr, words, check_aligned)?.as_ptr();
            let in_place = is_vector && input as usize == output as usize;
            if !in_place
                && !crate::mem_ops::is_nonoverlapping(
                    input as usize,
                    words as usize * size_of::<u64>(),
                    output as usize,
                    output_bytes,
                )
            {
                return Err(SyscallError::CopyOverlapping.into());
            }
            inputs[index] = input;
            strides[index] = if is_vector { words_per_lane as usize } else { 0 };
        }

        if sequence.wide {
            run_lanes::<u128>(&sequence, lane_count as usize, inputs, strides, output)?;
        } else {
            #[cfg(target_arch = "x86_64")]
            if avx512::available() {
                // SAFETY: the pointers were translated for every lane above and the CPU
                // features were just checked.
                unsafe { avx512::run_lanes_u64(&sequence, lane_count as usize, inputs, strides, output)? };
                return Ok(0);
            }
            run_lanes::<u64>(&sequence, lane_count as usize, inputs, strides, output)?;
        }
        Ok(0)
    }
}

fn run_lanes<L: Lane>(
    sequence: &OpSequence,
    lane_count: usize,
    inputs: [*const u64; 3],
    strides: [usize; 3],
    output: *mut u64,
) -> Result<(), Error> {
    for lane in 0..lane_count {
        // SAFETY: every pointer was translated for `lane_count` lanes (or one, when the stride
        // is zero) and the loop stays inside those bounds.
        let (a, b, c) = unsafe {
            (
                L::read(inputs[0].add(lane * strides[0])),
                L::read(inputs[1].add(lane * strides[1])),
                L::read(inputs[2].add(lane * strides[2])),
            )
        };
        let result = sequence.evaluate(a, b, c)?;
        unsafe { result.write(output.add(lane * L::WORDS)) };
    }
    Ok(())
}

/// The same stack program over eight u64 lanes at a time. Each token is one AVX-512
/// instruction on the whole chunk, so the token dispatch runs once per eight lanes and the
/// stack lives in `zmm` registers. Divides have no vector form and fall back to per-lane
/// scalar code after a vector zero check.
#[cfg(target_arch = "x86_64")]
mod avx512 {
    use {super::*, std::arch::x86_64::*};

    const LANES: usize = 8;

    pub fn available() -> bool {
        std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512ifma")
    }

    /// # Safety
    /// Every pointer must cover `lane_count` lanes (one lane where its stride is zero), and
    /// the CPU must support AVX-512F and AVX-512IFMA.
    #[target_feature(enable = "avx512f,avx512ifma")]
    pub unsafe fn run_lanes_u64(
        sequence: &OpSequence,
        lane_count: usize,
        inputs: [*const u64; 3],
        strides: [usize; 3],
        output: *mut u64,
    ) -> Result<(), Error> {
        let mut lane = 0;
        while lane < lane_count {
            let remaining = lane_count - lane;
            let active: __mmask8 = if remaining >= LANES { 0xff } else { (1u8 << remaining) - 1 };
            let (a, b, c) = unsafe {
                (
                    load_input(inputs[0], strides[0], lane, active),
                    load_input(inputs[1], strides[1], lane, active),
                    load_input(inputs[2], strides[2], lane, active),
                )
            };
            let result = unsafe { evaluate_chunk(sequence, a, b, c, active)? };
            unsafe { _mm512_mask_storeu_epi64(output.add(lane) as *mut i64, active, result) };
            lane += LANES;
        }
        Ok(())
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn load_input(pointer: *const u64, stride: usize, lane: usize, active: __mmask8) -> __m512i {
        if stride == 0 {
            _mm512_set1_epi64(unsafe { ptr::read_unaligned(pointer) } as i64)
        } else {
            unsafe { _mm512_maskz_loadu_epi64(active, pointer.add(lane) as *const i64) }
        }
    }

    #[target_feature(enable = "avx512f,avx512ifma")]
    unsafe fn evaluate_chunk(
        sequence: &OpSequence,
        a: __m512i,
        b: __m512i,
        c: __m512i,
        active: __mmask8,
    ) -> Result<__m512i, Error> {
        let zero = _mm512_setzero_si512();
        let all_ones = _mm512_set1_epi64(-1);
        let mut stack = [zero; MAX_TOKENS];
        let mut depth = 0usize;
        for token in &sequence.tokens[..sequence.len] {
            let base = depth - token.operands();
            let result = match *token {
                Token::PushA => a,
                Token::PushB => b,
                Token::PushC => c,
                Token::Lit(value) => _mm512_set1_epi64(value as i64),
                Token::Dup => stack[base],
                Token::Not => _mm512_xor_si512(stack[base], all_ones),
                Token::Neg => _mm512_sub_epi64(zero, stack[base]),
                Token::Add => _mm512_add_epi64(stack[base], stack[base + 1]),
                Token::Sub => _mm512_sub_epi64(stack[base], stack[base + 1]),
                Token::And => _mm512_and_si512(stack[base], stack[base + 1]),
                Token::Or => _mm512_or_si512(stack[base], stack[base + 1]),
                Token::Xor => _mm512_xor_si512(stack[base], stack[base + 1]),
                Token::Min => _mm512_min_epi64(stack[base], stack[base + 1]),
                Token::Max => _mm512_max_epi64(stack[base], stack[base + 1]),
                // variable shifts by 64 or more give 0 (sign fill for srav), matching `Lane`
                Token::Lsh => _mm512_sllv_epi64(stack[base], stack[base + 1]),
                Token::Lrsh => _mm512_srlv_epi64(stack[base], stack[base + 1]),
                Token::Arsh => _mm512_srav_epi64(stack[base], stack[base + 1]),
                Token::Div => unsafe { divide(stack[base], stack[base + 1], active, false)? },
                Token::Mod => unsafe { divide(stack[base], stack[base + 1], active, true)? },
                Token::MulLo52 => _mm512_madd52lo_epu64(zero, stack[base], stack[base + 1]),
                Token::MulHi52 => _mm512_madd52hi_epu64(zero, stack[base], stack[base + 1]),
                Token::Mul128Div => unsafe {
                    mul_wide_div(stack[base], stack[base + 1], stack[base + 2], active)?
                },
                Token::Mul128Shr => unsafe { mul_wide_shr(stack[base], stack[base + 1], stack[base + 2]) },
                Token::Select => _mm512_mask_blend_epi64(
                    _mm512_test_epi64_mask(stack[base], stack[base]),
                    stack[base + 2],
                    stack[base + 1],
                ),
            };
            stack[base] = result;
            depth = base + 1;
        }
        Ok(stack[0])
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn to_array(vector: __m512i) -> [u64; LANES] {
        let mut lanes = [0u64; LANES];
        unsafe { _mm512_storeu_epi64(lanes.as_mut_ptr() as *mut i64, vector) };
        lanes
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn from_array(lanes: &[u64; LANES]) -> __m512i {
        unsafe { _mm512_loadu_epi64(lanes.as_ptr() as *const i64) }
    }

    /// Fails if any active lane divides by zero.
    #[target_feature(enable = "avx512f")]
    unsafe fn check_divisors(divisors: __m512i, active: __mmask8) -> Result<(), Error> {
        let zero_divisor = _mm512_cmpeq_epi64_mask(divisors, _mm512_setzero_si512());
        if zero_divisor & active != 0 {
            return Err(SyscallError::MathDivideByZero.into());
        }
        Ok(())
    }

    /// There is no vector integer divide; per lane, after the zero check.
    #[target_feature(enable = "avx512f")]
    unsafe fn divide(
        dividends: __m512i,
        divisors: __m512i,
        active: __mmask8,
        remainder: bool,
    ) -> Result<__m512i, Error> {
        unsafe { check_divisors(divisors, active)? };
        let mut results = unsafe { to_array(dividends) };
        let divisors = unsafe { to_array(divisors) };
        for lane in 0..LANES {
            if divisors[lane] != 0 {
                results[lane] = if remainder {
                    results[lane] % divisors[lane]
                } else {
                    results[lane] / divisors[lane]
                };
            }
        }
        Ok(unsafe { from_array(&results) })
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn mul_wide_div(
        x: __m512i,
        y: __m512i,
        divisors: __m512i,
        active: __mmask8,
    ) -> Result<__m512i, Error> {
        unsafe { check_divisors(divisors, active)? };
        let mut results = unsafe { to_array(x) };
        let y = unsafe { to_array(y) };
        let divisors = unsafe { to_array(divisors) };
        for lane in 0..LANES {
            if divisors[lane] != 0 {
                results[lane] = ((results[lane] as u128 * y[lane] as u128) / divisors[lane] as u128) as u64;
            }
        }
        Ok(unsafe { from_array(&results) })
    }

    /// Full 64x64 -> 128 product per lane as `(high, low)`, from four 32x32 partial products.
    #[target_feature(enable = "avx512f")]
    unsafe fn mul_wide(x: __m512i, y: __m512i) -> (__m512i, __m512i) {
        let x_high = _mm512_srli_epi64::<32>(x);
        let y_high = _mm512_srli_epi64::<32>(y);
        let low_low = _mm512_mul_epu32(x, y);
        let low_high = _mm512_mul_epu32(x, y_high);
        let high_low = _mm512_mul_epu32(x_high, y);
        let high_high = _mm512_mul_epu32(x_high, y_high);
        let middle = _mm512_add_epi64(low_high, high_low);
        let middle_carry = _mm512_cmplt_epu64_mask(middle, low_high);
        let low = _mm512_add_epi64(low_low, _mm512_slli_epi64::<32>(middle));
        let low_carry = _mm512_cmplt_epu64_mask(low, low_low);
        let mut high = _mm512_add_epi64(high_high, _mm512_srli_epi64::<32>(middle));
        high = _mm512_mask_add_epi64(high, middle_carry, high, _mm512_set1_epi64(1i64 << 32));
        high = _mm512_mask_add_epi64(high, low_carry, high, _mm512_set1_epi64(1));
        (high, low)
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn mul_wide_shr(x: __m512i, y: __m512i, shift: __m512i) -> __m512i {
        let (high, low) = unsafe { mul_wide(x, y) };
        let sixty_four = _mm512_set1_epi64(64);
        let below_64 = _mm512_cmplt_epu64_mask(shift, sixty_four);
        let below_128 = _mm512_cmplt_epu64_mask(shift, _mm512_set1_epi64(128));
        // shift < 64: (low >> shift) | (high << (64 - shift)); a variable shift by 64 is 0,
        // so shift = 0 comes out right too
        let low_part = _mm512_or_si512(
            _mm512_srlv_epi64(low, shift),
            _mm512_sllv_epi64(high, _mm512_sub_epi64(sixty_four, shift)),
        );
        // 64 <= shift < 128: high >> (shift - 64); shift >= 128: 0
        let high_part = _mm512_maskz_mov_epi64(
            below_128,
            _mm512_srlv_epi64(high, _mm512_sub_epi64(shift, sixty_four)),
        );
        _mm512_mask_blend_epi64(below_64, high_part, low_part)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const VECTOR_ALL: u8 = 0b1110;

    #[test]
    fn decode_rejects_malformed_sequences() {
        assert!(OpSequence::decode(encode(0, &[])).is_err(), "empty");
        assert!(OpSequence::decode(encode(0, &[A, B])).is_err(), "two values left");
        assert!(OpSequence::decode(encode(0, &[ADD])).is_err(), "underflow");
        assert!(OpSequence::decode(encode(0, &[A, 63])).is_err(), "unknown token");
        assert!(OpSequence::decode(encode(0, &[A, B, C, A, B, C, A, B, C, LIT])).is_err(), "lit without value");
        assert_eq!(OpSequence::decode(encode(0, &[A, B, ADD])).unwrap().len(), 3);
    }

    #[test]
    fn lerp() {
        // t0 + (((t1 - t0) * f) >> 52)
        let word = encode(VECTOR_ALL, &[A, B, A, SUB, C, MUL_HI52, ADD]);
        let half = 1u64 << 51;
        assert_eq!(evaluate_u64(word, 100, 300, half).unwrap(), 200);
    }

    #[test]
    fn mul128_shr_matches_u128_arithmetic() {
        let word = encode(0, &[A, B, LIT, 26, MUL128_SHR]);
        let (a, b) = (0x1234_5678_9abc_def0u64, 0x0fed_cba9_8765_4321u64);
        assert_eq!(evaluate_u64(word, a, b, 0).unwrap(), ((a as u128 * b as u128) >> 26) as u64);
        let wide = encode(FLAG_WIDE as u8, &[A, B, LIT, 60, MUL128_SHR]);
        let (a, b) = (u128::MAX / 3, 0xffff_ffff_ffff_ffffu128 << 7);
        let expected = {
            let (high, low) = mul_wide_u128(a, b);
            (low >> 60) | (high << 68)
        };
        assert_eq!(evaluate_u128(wide, a, b, 0).unwrap(), expected);
    }

    #[test]
    fn mul128_div_and_wide_division() {
        let word = encode(0, &[A, B, C, MUL128_DIV]);
        assert_eq!(evaluate_u64(word, u64::MAX, 1_000_000_007, 1_000_000_007).unwrap(), u64::MAX);
        let (high, low) = mul_wide_u128(u128::MAX, u128::MAX - 5);
        assert_eq!(div_wide_u128(high, low, u128::MAX), u128::MAX - 5);
        assert_eq!(div_wide_u128(0, 1_000, 7), 142);
        assert!(evaluate_u64(word, 1, 2, 0).is_err(), "divide by zero");
    }

    #[test]
    fn select_without_a_compare() {
        // (-x) >> 63 is all ones exactly when x > 0:  x > 0 ? b : c
        let word = encode(0, &[A, NEG, LIT, 63, ARSH, B, C, SELECT]);
        assert_eq!(evaluate_u64(word, 5, 10, 20).unwrap(), 10);
        assert_eq!(evaluate_u64(word, 0, 10, 20).unwrap(), 20);
        assert_eq!(evaluate_u64(word, (-5i64) as u64, 10, 20).unwrap(), 20);
    }

    #[test]
    fn shifts_past_the_width_and_signed_min_max() {
        assert_eq!(evaluate_u64(encode(0, &[A, LIT, 63, LSH]), 1, 0, 0).unwrap(), 1 << 63);
        assert_eq!(evaluate_u64(encode(0, &[A, B, LSH]), 1, 64, 0).unwrap(), 0);
        assert_eq!(evaluate_u64(encode(0, &[A, B, ARSH]), u64::MAX, 64, 0).unwrap(), u64::MAX);
        assert_eq!(evaluate_u64(encode(0, &[A, B, MAX]), (-1i64) as u64, 1, 0).unwrap(), 1);
        assert_eq!(evaluate_u64(encode(0, &[A, B, MIN]), (-1i64) as u64, 1, 0).unwrap(), (-1i64) as u64);
    }

    /// Every op, vector and broadcast inputs, full and partial chunks: AVX-512 against scalar.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx512_matches_scalar() {
        if !avx512::available() {
            eprintln!("skipping: this CPU has no AVX-512 IFMA");
            return;
        }
        let sequences: [&[u8]; 22] = [
            &[A, B, ADD], &[A, B, SUB], &[A, B, AND], &[A, B, OR], &[A, B, XOR],
            &[A, NOT], &[A, NEG], &[A, DUP, ADD], &[A, B, MIN], &[A, B, MAX],
            &[A, B, LSH], &[A, B, LRSH], &[A, B, ARSH], &[A, B, DIV], &[A, B, MOD],
            &[A, B, MUL_LO52], &[A, B, MUL_HI52], &[A, B, C, MUL128_SHR], &[A, B, C, MUL128_DIV],
            &[A, B, C, SELECT], &[A, B, A, SUB, C, MUL_HI52, ADD], &[A, LIT, 26, LRSH, B, MIN],
        ];
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for tokens in sequences {
            let sequence = OpSequence::decode(encode(VECTOR_ALL, tokens)).unwrap();
            for lane_count in [1usize, 7, 8, 9, 64] {
                let a: Vec<u64> = (0..lane_count).map(|_| next()).collect();
                // small values in some lanes so shifts, divisors and select conditions vary
                let b: Vec<u64> = (0..lane_count).map(|lane| if lane % 3 == 0 { next() % 70 + 1 } else { next() | 1 }).collect();
                let c: Vec<u64> = (0..lane_count).map(|lane| if lane % 2 == 0 { next() % 140 + 1 } else { next() | 1 }).collect();
                for strides in [[1usize, 1, 1], [0, 1, 1], [1, 0, 1]] {
                    let expected: Vec<u64> = (0..lane_count)
                        .map(|lane| {
                            let pick = |values: &[u64], stride: usize| values[lane * stride];
                            sequence.evaluate(pick(&a, strides[0]), pick(&b, strides[1]), pick(&c, strides[2])).unwrap()
                        })
                        .collect();
                    let mut output = vec![0u64; lane_count];
                    unsafe {
                        avx512::run_lanes_u64(
                            &sequence,
                            lane_count,
                            [a.as_ptr(), b.as_ptr(), c.as_ptr()],
                            strides,
                            output.as_mut_ptr(),
                        )
                        .unwrap()
                    };
                    assert_eq!(output, expected, "{tokens:?} lanes {lane_count} strides {strides:?}");
                }
            }
        }
    }
}
