//! Bounded structural preflight before allocating a decoded NBT tree.
use super::*;
use serde::de::DeserializeOwned;
pub(crate) fn decode<T: DeserializeOwned>(bytes: &[u8]) -> WorldResult<T> {
    if bytes.len() > 64 * 1024 * 1024 {
        return Err(WorldError::Invalid("nbt_payload_limit"));
    }
    validate_wire(bytes)?;
    fastnbt::from_bytes_with_opts(bytes, fastnbt::DeOpts::new().max_seq_len(2_000_000))
        .map_err(|e| WorldError::Codec(e.to_string()))
}
pub(crate) fn validate_wire(bytes: &[u8]) -> WorldResult<()> {
    if bytes.len() > 64 * 1024 * 1024 {
        return Err(WorldError::Invalid("nbt_payload_limit"));
    }
    let mut scanner = Scanner {
        bytes,
        offset: 0,
        budget: 2_000_000,
    };
    if scanner.byte()? != 10 {
        return Err(WorldError::Invalid("nbt_root_compound"));
    }
    scanner.string()?;
    scanner.payload(10, 0)?;
    if scanner.offset != bytes.len() {
        return Err(WorldError::Invalid("nbt_trailing_data"));
    }
    Ok(())
}
struct Scanner<'a> {
    bytes: &'a [u8],
    offset: usize,
    budget: usize,
}
impl<'a> Scanner<'a> {
    fn take(&mut self, count: usize) -> WorldResult<&'a [u8]> {
        let end = self
            .offset
            .checked_add(count)
            .ok_or(WorldError::Invalid("nbt_size"))?;
        let bytes = self
            .bytes
            .get(self.offset..end)
            .ok_or(WorldError::Invalid("nbt_truncated"))?;
        self.offset = end;
        Ok(bytes)
    }
    fn byte(&mut self) -> WorldResult<u8> {
        Ok(self.take(1)?[0])
    }
    fn string(&mut self) -> WorldResult<&'a [u8]> {
        let bytes = self.take(2)?;
        let count = u16::from_be_bytes([bytes[0], bytes[1]]) as usize;
        self.take(count)
    }
    fn count(&mut self) -> WorldResult<usize> {
        let count = i32::from_be_bytes(self.take(4)?.try_into().unwrap());
        let count =
            usize::try_from(count).map_err(|_| WorldError::Invalid("nbt_negative_length"))?;
        if count > 2_000_000 {
            return Err(WorldError::Invalid("nbt_sequence_budget"));
        }
        Ok(count)
    }
    fn payload(&mut self, tag: u8, depth: usize) -> WorldResult<()> {
        if depth > 32 || self.budget == 0 {
            return Err(WorldError::Invalid("nbt_depth_or_size"));
        }
        self.budget -= 1;
        match tag {
            1 => {
                self.take(1)?;
            }
            2 => {
                self.take(2)?;
            }
            3 | 5 => {
                self.take(4)?;
            }
            4 | 6 => {
                self.take(8)?;
            }
            8 => {
                self.string()?;
            }
            7 | 11 | 12 => {
                let count = self.count()?;
                let size = match tag {
                    7 => 1,
                    11 => 4,
                    _ => 8,
                };
                self.take(
                    count
                        .checked_mul(size)
                        .ok_or(WorldError::Invalid("nbt_array_size"))?,
                )?;
            }
            9 => {
                let element = self.byte()?;
                let count = self.count()?;
                self.budget = self
                    .budget
                    .checked_sub(count)
                    .ok_or(WorldError::Invalid("nbt_sequence_budget"))?;
                if count > 0 && element == 0 {
                    return Err(WorldError::Invalid("nbt_list_end_type"));
                }
                for _ in 0..count {
                    self.payload(element, depth + 1)?;
                }
            }
            10 => {
                let mut keys = std::collections::BTreeSet::new();
                loop {
                    let element = self.byte()?;
                    if element == 0 {
                        break;
                    }
                    let name = self.string()?;
                    if !keys.insert(name) {
                        return Err(WorldError::Invalid("nbt_duplicate_key"));
                    }
                    self.payload(element, depth + 1)?;
                }
            }
            _ => return Err(WorldError::Invalid("nbt_tag")),
        }
        Ok(())
    }
}
