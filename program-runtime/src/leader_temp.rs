use {
    serde::{Deserialize, Serialize},
    solana_pubkey::Pubkey,
    solana_sysvar_id::impl_sysvar_id,
};

pub const LEADER_INFO_ID: Pubkey =
    Pubkey::from_str_const("SysvarLeader1111111111111111111111111111111");

#[repr(C)]
#[derive(Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LeaderInfo {
    pub leader_id: Pubkey,
    pub next_leader_id: Pubkey,
    pub leader_vote: Pubkey,
    pub next_leader_vote: Pubkey,
}

fn id() -> Pubkey {
    LEADER_INFO_ID
}

fn check_id(address: &Pubkey) -> bool {
    *address == LEADER_INFO_ID
}

impl_sysvar_id!(LeaderInfo);
