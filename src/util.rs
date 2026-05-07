use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

pub fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max { return s.to_string(); }
    let mut end = max.saturating_sub(1);
    while !s.is_char_boundary(end) { end -= 1; }
    format!("{}…", &s[..end])
}

/// Truncate `s` to at most `max_cols` terminal display columns.
/// Accounts for wide (CJK) characters that occupy 2 columns each.
pub fn truncate_display(s: &str, max_cols: usize) -> String {
    if s.width() <= max_cols { return s.to_string(); }
    let mut cols = 0usize;
    let mut end = 0usize;
    for ch in s.chars() {
        let w = UnicodeWidthChar::width(ch).unwrap_or(1);
        if cols + w > max_cols.saturating_sub(1) { break; }
        cols += w;
        end += ch.len_utf8();
    }
    format!("{}…", &s[..end])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncate_within_limit() {
        assert_eq!(truncate("hello", 10), "hello");
    }

    #[test]
    fn truncate_at_limit() {
        assert_eq!(truncate("hello", 5), "hello");
    }

    #[test]
    fn truncate_over_limit() {
        let r = truncate("hello world", 7);
        assert!(r.ends_with('…'));
        assert!(r.len() <= 7 + 3); // "…" is 3 bytes
    }

    #[test]
    fn truncate_utf8_boundary() {
        // "café" = 6 bytes (c=1, a=1, f=1, é=2). Truncating at 4 must not panic.
        let s = truncate("café world", 4);
        assert!(s.is_char_boundary(s.len().saturating_sub(3))); // ends with valid "…"
    }

    #[test]
    fn truncate_empty() {
        assert_eq!(truncate("", 5), "");
    }

    #[test]
    fn truncate_display_cjk() {
        // Each CJK char is 2 display columns; 6 chars * 2 = 12 cols, limit to 8
        let s = truncate_display("项目名称很长", 8);
        let width: usize = s.chars()
            .map(|c| UnicodeWidthChar::width(c).unwrap_or(1))
            .sum();
        assert!(width <= 8, "display width {width} exceeds 8");
    }

    #[test]
    fn truncate_display_ascii() {
        let s = truncate_display("hello world", 7);
        assert!(s.ends_with('…'));
        assert!(s.width() <= 7);
    }

    #[test]
    fn truncate_display_fits() {
        assert_eq!(truncate_display("hello", 10), "hello");
    }
}
