"""Date parsing utilities and constants for email processing."""

# Timezone name to offset mapping
TZ_MAP = {
    'EDT': '-0400',  # Eastern Daylight Time
    'EST': '-0500',  # Eastern Standard Time
    'CDT': '-0500',  # Central Daylight Time
    'CST': '-0600',  # Central Standard Time
    'MDT': '-0600',  # Mountain Daylight Time
    'MST': '-0700',  # Mountain Standard Time
    'PDT': '-0700',  # Pacific Daylight Time
    'PST': '-0800',  # Pacific Standard Time
    'IST': '+0530',  # Indian Standard Time
}

# List of date formats to try, in order of preference
DATE_FORMATS = [
    # RFC 2822 and common variants
    '%a, %d %b %Y %H:%M:%S %z',  # Standard email format
    '%d %b %Y %H:%M:%S %z',      # Without weekday
    '%a, %d %b %Y %H:%M:%S %Z',  # With timezone name
    '%a, %d %b %Y %H:%M:%S',     # Without timezone
    '%a, %d %b %Y %H:%M:%S %z',  # Without seconds
    '%d %b %Y %H:%M:%S %z',      # Without seconds and weekday
    
    # Unix style formats
    '%a %b %d %H:%M:%S %Y %z',   # With timezone
    '%a %b %d %H:%M:%S %Y',      # Without timezone
    '%a %b %d %H:%M:%S %z %Y',   # Timezone before year
    
    # Short year formats
    '%d %b %y %H:%M:%S',         # Basic short year
    '%d %b %y %H:%M:%S %z',      # Short year with timezone
    '%a, %d %b %y %H:%M:%S %z',  # Full format with short year
    '%a, %d %b %y %H:%M:%S',     # Without timezone
    
    # ISO formats
    '%Y-%m-%d %H:%M:%S%z',       # With timezone
    '%Y-%m-%d %H:%M:%S',         # Without timezone
    
    # Extra space variations
    '%a,  %d %b %Y %H:%M:%S %z',  # Double space after comma
    '%a,  %d %b %y %H:%M:%S %z',  # Double space with short year
    '%a,  %d %b %Y %H:%M:%S',     # Double space without timezone
    '%a,  %d %b %y %H:%M:%S',     # Double space, short year, no timezone
    
    # Special formats
    '%a %b %d %H:%M:%S GMT%z %Y',  # GMT with offset
    '%a %b %d %H:%M:%S %z %Y',     # Offset before year
]

import re
from datetime import datetime, timezone
from functools import lru_cache

@lru_cache(maxsize=10000)
def parse_email_date(date_str: str) -> str:
    """Parse email date string into a consistent format.
    
    Handles various email date formats and converts to our standard format.
    
    Args:
        date_str: Date string from email header
        
    Returns:
        Date string in format 'YYYY-MM-DD HH:MM:SS±HHMM'
    """
    if not date_str:
        raise ValueError("Empty date string")
    
    # Try to handle common variations
    date_str = date_str.strip()
    
    # Clean up parenthetical timezone names
    date_str = re.sub(r'\s*\([^)]+\)\s*$', '', date_str)
    
    # Handle Unix-style timestamps with timezone but no space
    # e.g., "Thu Apr 16 20:59:04 2015+0530" -> "Thu Apr 16 20:59:04 2015 +0530"
    unix_tz_match = re.search(r'(\d{4})[+-]\d{4}$', date_str)
    if unix_tz_match:
        year_pos = date_str.find(unix_tz_match.group(1))
        if year_pos != -1:
            year_end = year_pos + 4
            date_str = date_str[:year_end] + ' ' + date_str[year_end:]
    
    # Handle UT timezone
    if ' UT' in date_str:
        date_str = date_str.replace(' UT', ' +0000')
    
    # Handle GMT+HHMM format
    if 'GMT+' in date_str:
        parts = date_str.split('GMT+')
        if len(parts) == 2:
            date_str = parts[0] + '+' + parts[1]
    
    # Replace timezone names with their offsets
    for tz_name, offset in TZ_MAP.items():
        if f" {tz_name}" in date_str:
            date_str = date_str.replace(f" {tz_name}", f" {offset}")
            break
    
    # Handle non-standard timezone offsets
    if (match := re.search(r'([+-])(\d{2}):?(\d{2})$', date_str)):
        sign, hours, mins = match.groups()
        if int(mins) > 30:
            hours = str(int(hours) + 1).zfill(2)
            mins = "00"
        else:
            mins = "30"
        date_str = re.sub(r'[+-]\d{2}:?\d{2}$', f'{sign}{hours}{mins}', date_str)
    
    # Try each format
    for fmt in DATE_FORMATS:
        try:
            # Parse with current format
            dt = datetime.strptime(date_str, fmt)
            
            # If timezone is naive, assume UTC
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            
            # Format to our standard format
            return dt.strftime('%Y-%m-%d %H:%M:%S%z')
            
        except ValueError:
            continue
            
    # If we get here, none of our formats worked
    raise ValueError(f"Could not parse date string: {date_str}")

def parse_date_for_sorting(date_str: str) -> datetime:
    """Parse date string into datetime object for sorting. Returns None if parsing fails."""
    if not date_str:
        return None

    try:
        # Try our standard format first
        return datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S%z')
    except ValueError:
        # If that fails, try to parse it with parse_email_date and then convert
        try:
            standardized = parse_email_date(date_str)
            return datetime.strptime(standardized, '%Y-%m-%d %H:%M:%S%z')
        except ValueError:
            return None
