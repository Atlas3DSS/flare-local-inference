use crate::{Result, model::VOCABULARY};

pub(super) struct TokenStops {
    sequences: Vec<Vec<u32>>,
    held: Vec<u32>,
}
impl TokenStops {
    pub fn new(sequences: Vec<Vec<u32>>) -> Result<Self> {
        if sequences.len() > 16
            || sequences.iter().any(|sequence| {
                sequence.is_empty()
                    || sequence.len() > 64
                    || sequence.iter().any(|token| *token as usize >= VOCABULARY)
            })
        {
            return Err(
                "token stops allow at most sixteen nonempty sequences of at most 64 vocabulary IDs"
                    .into(),
            );
        }
        Ok(Self {
            sequences,
            held: Vec::with_capacity(64),
        })
    }
    pub fn push(&mut self, token: u32) -> (Vec<u32>, bool) {
        self.held.push(token);
        if let Some(length) = self
            .sequences
            .iter()
            .filter(|sequence| self.held.ends_with(sequence))
            .map(Vec::len)
            .max()
        {
            let visible = self.held.drain(..self.held.len() - length).collect();
            self.held.clear();
            return (visible, true);
        }
        let retain = self
            .sequences
            .iter()
            .flat_map(|sequence| {
                (1..sequence.len())
                    .rev()
                    .filter(|length| self.held.ends_with(&sequence[..*length]))
                    .take(1)
            })
            .max()
            .unwrap_or(0);
        (self.held.drain(..self.held.len() - retain).collect(), false)
    }
    pub fn finish(&mut self) -> Vec<u32> {
        std::mem::take(&mut self.held)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn overlapping_stop_prefixes_are_held_without_losing_ordinary_tokens() {
        let mut stops = TokenStops::new(vec![vec![1, 2, 3], vec![2, 4]]).unwrap();
        assert_eq!(stops.push(9), (vec![9], false));
        assert_eq!(stops.push(1), (vec![], false));
        assert_eq!(stops.push(2), (vec![], false));
        assert_eq!(stops.push(4), (vec![1], true));
        assert!(stops.finish().is_empty());
    }
    #[test]
    fn terminal_flush_preserves_an_unfinished_stop_prefix() {
        let mut stops = TokenStops::new(vec![vec![1, 2, 3]]).unwrap();
        assert_eq!(stops.push(1), (vec![], false));
        assert_eq!(stops.finish(), vec![1]);
    }
}
