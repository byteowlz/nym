//! Duplicate-rejecting JSON maps shared by decision responses and corpus inventory.
use serde::{
    Deserialize,
    de::{MapAccess, Visitor},
};

pub(crate) fn members<'de, D, T>(deserializer: D) -> Result<Vec<(String, T)>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de>,
{
    struct Unique<T>(std::marker::PhantomData<T>);
    impl<'de, T: Deserialize<'de>> Visitor<'de> for Unique<T> {
        type Value = Vec<(String, T)>;
        fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("map with unique keys")
        }
        fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
            let mut rows = Vec::new();
            let mut seen = std::collections::HashSet::new();
            while let Some((key, value)) = map.next_entry::<String, T>()? {
                if !seen.insert(key.clone()) {
                    return Err(serde::de::Error::custom("duplicate JSON-map key"));
                }
                rows.push((key, value));
            }
            Ok(rows)
        }
    }
    deserializer.deserialize_map(Unique(std::marker::PhantomData))
}
#[cfg(feature = "decision")]
pub(crate) fn map<'de, D, T>(
    deserializer: D,
) -> Result<std::collections::HashMap<String, T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de>,
{
    members(deserializer).map(|items| items.into_iter().collect())
}
