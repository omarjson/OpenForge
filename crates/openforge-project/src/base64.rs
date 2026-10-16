//! Standard base64 (RFC 4648, `=`-padded) for texture pixels.
//!
//! Texture pixels are bytes, not a shape a JSON array should carry: a
//! 256×256 texture would become a quarter-million-element number array, and a
//! project file is meant to stay plain, readable text. Base64 keeps it text
//! and about a third larger than the raw pixels instead of several times
//! larger — and it is the one encoding both a reader and a writer can agree
//! on without a new binary section in the format.

const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Encode `bytes` as standard base64 with `=` padding.
pub fn encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b0 = chunk[0];
        let b1 = *chunk.get(1).unwrap_or(&0);
        let b2 = *chunk.get(2).unwrap_or(&0);
        let n = (u32::from(b0) << 16) | (u32::from(b1) << 8) | u32::from(b2);
        out.push(ALPHABET[(n >> 18) as usize & 63] as char);
        out.push(ALPHABET[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 {
            ALPHABET[(n >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            ALPHABET[n as usize & 63] as char
        } else {
            '='
        });
    }
    out
}

/// Decode standard base64; `None` for anything that is not well-formed.
///
/// Strict on purpose: wrong lengths, padding anywhere but at the very end,
/// characters outside the alphabet and data after a `=` are all refused, so a
/// texture that comes back from a project file is either exactly what went in
/// or the file is reported as malformed.
pub fn decode(text: &str) -> Option<Vec<u8>> {
    let bytes = text.as_bytes();
    if bytes.len() % 4 != 0 {
        return None;
    }
    let chunks = bytes.len() / 4;
    let mut out = Vec::with_capacity(chunks * 3);
    for (index, chunk) in bytes.chunks(4).enumerate() {
        let mut values = [0u32; 4];
        let mut padding = 0usize;
        for (i, &c) in chunk.iter().enumerate() {
            match c {
                b'A'..=b'Z' => values[i] = u32::from(c - b'A'),
                b'a'..=b'z' => values[i] = u32::from(c - b'a') + 26,
                b'0'..=b'9' => values[i] = u32::from(c - b'0') + 52,
                b'+' => values[i] = 62,
                b'/' => values[i] = 63,
                b'=' if i >= 2 => padding += 1,
                _ => return None,
            }
            if padding > 0 && c != b'=' {
                return None;
            }
        }
        if padding > 0 && index + 1 != chunks {
            return None;
        }
        let n = (values[0] << 18) | (values[1] << 12) | (values[2] << 6) | values[3];
        out.push((n >> 16) as u8);
        if padding < 2 {
            out.push((n >> 8) as u8);
        }
        if padding < 1 {
            out.push(n as u8);
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_rfc_vectors_encode_and_decode() {
        for (text, encoded) in [
            ("", ""),
            ("f", "Zg=="),
            ("fo", "Zm8="),
            ("foo", "Zm9v"),
            ("foob", "Zm9vYg=="),
            ("fooba", "Zm9vYmE="),
            ("foobar", "Zm9vYmFy"),
        ] {
            assert_eq!(encode(text.as_bytes()), encoded, "encoding {text:?}");
            assert_eq!(
                decode(encoded).as_deref(),
                Some(text.as_bytes()),
                "decoding {encoded:?}"
            );
        }
    }

    #[test]
    fn every_byte_round_trips() {
        let all: Vec<u8> = (0..=255).collect();
        // A length that exercises every padding case at once.
        assert_eq!(decode(&encode(&all)).as_deref(), Some(all.as_slice()));
        for len in 0..=8 {
            let slice = &all[..len];
            assert_eq!(decode(&encode(slice)).as_deref(), Some(slice));
        }
    }

    #[test]
    fn malformed_text_is_refused() {
        for text in [
            "A",        // length not a multiple of four
            "AAAAA",    // ditto
            "====",     // padding everywhere
            "A===",     // padding before a data position
            "AA=A",     // data after padding
            "=AAA",     // padding first
            "AA==AA==", // padding in a non-final chunk
            "!!!!",     // outside the alphabet
            "AA**",     // ditto
            "AA A",     // whitespace is not part of the alphabet
            "AAAA=AAA",
        ] {
            assert_eq!(decode(text), None, "{text:?} should not decode");
        }
    }
}
