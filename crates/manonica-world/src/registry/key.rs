use crate::{WorldError, WorldResult};
use serde::{Deserialize, Serialize};
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct ResourceKey {
    namespace: String,
    path: String,
}
impl ResourceKey {
    pub fn parse(text: &str) -> WorldResult<Self> {
        let (namespace, path) = text
            .split_once(':')
            .ok_or(WorldError::Invalid("resource_namespace"))?;
        let valid = |c: char| c.is_ascii_lowercase() || c.is_ascii_digit() || "_.-".contains(c);
        if text.len() > 256
            || namespace.is_empty()
            || path.is_empty()
            || !namespace.chars().all(valid)
            || !path.chars().all(|c| valid(c) || c == '/')
        {
            return Err(WorldError::Invalid("resource_key"));
        }
        Ok(Self {
            namespace: namespace.into(),
            path: path.into(),
        })
    }
    pub fn namespace(&self) -> &str {
        &self.namespace
    }
    pub fn path(&self) -> &str {
        &self.path
    }
}
impl std::fmt::Display for ResourceKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.namespace, self.path)
    }
}
impl TryFrom<String> for ResourceKey {
    type Error = WorldError;
    fn try_from(v: String) -> WorldResult<Self> {
        Self::parse(&v)
    }
}
impl From<ResourceKey> for String {
    fn from(v: ResourceKey) -> Self {
        format!("{}:{}", v.namespace, v.path)
    }
}
