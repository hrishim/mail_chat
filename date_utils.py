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
    '%a, %d %b %Y %H:%M %z',     # Without seconds
    '%d %b %Y %H:%M %z',         # Without seconds and weekday
    
    # Unix style formats
    '%a %b %d %H:%M:%S %Y %z',   # With timezone
    '%a %b %d %H:%M:%S %Y',      # Without timezone
    '%a %b %d %H:%M:%S %z %Y',   # Timezone before year
    
    # Short year formats
    '%d %b %y %H:%M:%S',         # Basic short year
    '%d %b %y %H:%M %z',         # Short year with timezone
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
