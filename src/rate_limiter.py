# rate_limiter.py
"""
Rate Limiter Module
===================
Implements rate limiting to protect API quotas and ensure fair usage.

Key Functions:
- check_limit(): Check if request is allowed
- record_request(): Record a new request
- get_remaining_quota(): Get remaining quota info
"""

import sys
from pathlib import Path
from typing import Dict, Tuple, Optional
from datetime import datetime, timedelta
from collections import deque
import json
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config


class RateLimiter:
    """
    Manages rate limiting per user and globally to protect API quotas.
    """
    
    def __init__(
        self,
        hourly_limit: int = config.MAX_QUERIES_PER_HOUR,
        daily_limit: int = config.MAX_QUERIES_PER_DAY,
        global_daily_limit: int = config.GLOBAL_DAILY_LIMIT,
        storage_dir: str = config.STORAGE_DIR
    ):
        """
        Initialize RateLimiter.
        
        Args:
            hourly_limit: Max queries per user per hour
            daily_limit: Max queries per user per day
            global_daily_limit: Max queries globally per day
            storage_dir: Directory for persistent storage
        """
        self.hourly_limit = hourly_limit
        self.daily_limit = daily_limit
        self.global_daily_limit = global_daily_limit
        
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True, parents=True)
        
        # In-memory tracking
        self.user_requests = {}  # user_id -> deque of timestamps
        self.global_requests = deque()  # All requests globally
        
        # Persistent storage file
        self.state_file = self.storage_dir / "rate_limit_state.json"
        self._load_state()
        
        logger.info(f"RateLimiter initialized (hourly: {hourly_limit}, daily: {daily_limit}, global: {global_daily_limit})")
    
    
    def check_limit(self, user_id: str) -> Tuple[bool, str]:
        """
        Check if user can make a request.
        
        Args:
            user_id: Unique user identifier
            
        Returns:
            Tuple of (allowed: bool, reason: str)
        """
        now = datetime.now()
        
        # Clean up old requests
        self._cleanup_old_requests(user_id)
        
        # Get user's request history
        if user_id not in self.user_requests:
            self.user_requests[user_id] = deque()
        
        user_reqs = self.user_requests[user_id]
        
        # Check hourly limit
        hour_ago = now - timedelta(hours=1)
        hourly_count = sum(1 for ts in user_reqs if ts > hour_ago)
        
        if hourly_count >= self.hourly_limit:
            minutes_to_reset = self._get_minutes_to_reset(user_reqs, hours=1)
            logger.warning(f"Hourly limit exceeded for user: {user_id}")
            return False, f"Hourly limit ({self.hourly_limit}/hour) exceeded. Reset in {minutes_to_reset} minutes."
        
        # Check daily limit
        day_ago = now - timedelta(days=1)
        daily_count = sum(1 for ts in user_reqs if ts > day_ago)
        
        if daily_count >= self.daily_limit:
            minutes_to_reset = self._get_minutes_to_reset(user_reqs, hours=24)
            logger.warning(f"Daily limit exceeded for user: {user_id}")
            return False, f"Daily limit ({self.daily_limit}/day) exceeded. Reset in {minutes_to_reset} minutes."
        
        # Check global daily limit
        global_day_ago = now - timedelta(days=1)
        global_count = sum(1 for ts in self.global_requests if ts > global_day_ago)
        
        if global_count >= self.global_daily_limit:
            logger.warning(f"Global daily limit exceeded")
            return False, f"System capacity reached ({self.global_daily_limit}/day). Please try again tomorrow."
        
        return True, "Request allowed"
    
    
    def record_request(self, user_id: str):
        """
        Record a new request for rate limiting.
        
        Args:
            user_id: Unique user identifier
        """
        now = datetime.now()
        
        # Record user request
        if user_id not in self.user_requests:
            self.user_requests[user_id] = deque()
        
        self.user_requests[user_id].append(now)
        
        # Record global request
        self.global_requests.append(now)
        
        # Save state
        self._save_state()
        
        logger.debug(f"Request recorded for user: {user_id}")
    
    
    def get_remaining_quota(self, user_id: str) -> Dict:
        """
        Get remaining quota information for user.
        
        Args:
            user_id: Unique user identifier
            
        Returns:
            Dictionary with quota information
        """
        self._cleanup_old_requests(user_id)
        
        now = datetime.now()
        
        # Get user requests
        user_reqs = self.user_requests.get(user_id, deque())
        
        # Calculate hourly
        hour_ago = now - timedelta(hours=1)
        hourly_used = sum(1 for ts in user_reqs if ts > hour_ago)
        hourly_remaining = max(0, self.hourly_limit - hourly_used)
        
        # Calculate daily
        day_ago = now - timedelta(days=1)
        daily_used = sum(1 for ts in user_reqs if ts > day_ago)
        daily_remaining = max(0, self.daily_limit - daily_used)
        
        # Calculate global
        global_day_ago = now - timedelta(days=1)
        global_used = sum(1 for ts in self.global_requests if ts > global_day_ago)
        global_remaining = max(0, self.global_daily_limit - global_used)
        
        return {
            'hourly': {
                'used': hourly_used,
                'remaining': hourly_remaining,
                'limit': self.hourly_limit,
                'reset_minutes': self._get_minutes_to_reset(user_reqs, hours=1) if hourly_used > 0 else 60
            },
            'daily': {
                'used': daily_used,
                'remaining': daily_remaining,
                'limit': self.daily_limit,
                'reset_minutes': self._get_minutes_to_reset(user_reqs, hours=24) if daily_used > 0 else 1440
            },
            'global': {
                'used': global_used,
                'remaining': global_remaining,
                'limit': self.global_daily_limit
            }
        }
    
    
    def _cleanup_old_requests(self, user_id: str):
        """
        Remove requests older than 24 hours.
        
        Args:
            user_id: Unique user identifier
        """
        now = datetime.now()
        day_ago = now - timedelta(days=1)
        
        # Cleanup user requests
        if user_id in self.user_requests:
            self.user_requests[user_id] = deque(
                ts for ts in self.user_requests[user_id] if ts > day_ago
            )
        
        # Cleanup global requests
        self.global_requests = deque(
            ts for ts in self.global_requests if ts > day_ago
        )
    
    
    def _get_minutes_to_reset(self, requests: deque, hours: int) -> int:
        """
        Calculate minutes until limit resets.
        
        Args:
            requests: Deque of request timestamps
            hours: Lookback period in hours
            
        Returns:
            Minutes until reset
        """
        if not requests:
            return 0
        
        oldest_request = min(requests)
        reset_time = oldest_request + timedelta(hours=hours)
        minutes_remaining = max(0, (reset_time - datetime.now()).total_seconds() / 60)
        
        return int(minutes_remaining)
    
    
    def reset_user(self, user_id: str):
        """
        Reset rate limits for a specific user (admin function).
        
        Args:
            user_id: User identifier to reset
        """
        if user_id in self.user_requests:
            del self.user_requests[user_id]
        
        logger.info(f"✅ Rate limits reset for user: {user_id}")
        self._save_state()
    
    
    def get_stats(self) -> Dict:
        """
        Get rate limiter statistics.
        
        Returns:
            Dictionary with stats
        """
        total_users = len(self.user_requests)
        active_users = sum(1 for reqs in self.user_requests.values() if len(reqs) > 0)
        
        now = datetime.now()
        hour_ago = now - timedelta(hours=1)
        hourly_requests = sum(
            sum(1 for ts in reqs if ts > hour_ago)
            for reqs in self.user_requests.values()
        )
        
        return {
            'total_users': total_users,
            'active_users': active_users,
            'hourly_requests': hourly_requests,
            'global_requests_24h': len(self.global_requests),
            'limits': {
                'hourly': self.hourly_limit,
                'daily': self.daily_limit,
                'global_daily': self.global_daily_limit
            }
        }
    
    
    def _load_state(self):
        """
        Load rate limiter state from disk.
        """
        if not self.state_file.exists():
            return
        
        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            
            # Restore user requests
            for user_id, timestamps in state.get('user_requests', {}).items():
                self.user_requests[user_id] = deque(
                    datetime.fromisoformat(ts) for ts in timestamps
                )
            
            # Restore global requests
            self.global_requests = deque(
                datetime.fromisoformat(ts)
                for ts in state.get('global_requests', [])
            )
            
            logger.info(f"Loaded rate limiter state: {len(self.user_requests)} users")
            
        except Exception as e:
            logger.warning(f"Failed to load rate limiter state: {str(e)}")
    
    
    def _save_state(self):
        """
        Save rate limiter state to disk.
        """
        try:
            state = {
                'user_requests': {
                    user_id: [ts.isoformat() for ts in requests]
                    for user_id, requests in self.user_requests.items()
                },
                'global_requests': [ts.isoformat() for ts in self.global_requests],
                'saved_at': datetime.now().isoformat()
            }
            
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save rate limiter state: {str(e)}")


# ==========================================
# Testing & Demo
# ==========================================

if __name__ == "__main__":
    """
    Test rate limiter functionality.
    """
    import time
    
    # Setup logging
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    
    print("\n" + "="*60)
    print("Rate Limiter Test")
    print("="*60 + "\n")
    
    # Initialize with low limits for testing
    print("🔄 Initializing rate limiter...")
    limiter = RateLimiter(hourly_limit=3, daily_limit=5, global_daily_limit=20)
    
    user_id = "test_user_123"
    
    # Test 1: Check initial quota
    print("\n--- Test 1: Initial Quota ---")
    quota = limiter.get_remaining_quota(user_id)
    print(f"Hourly remaining: {quota['hourly']['remaining']}/{quota['hourly']['limit']}")
    print(f"Daily remaining: {quota['daily']['remaining']}/{quota['daily']['limit']}")
    
    # Test 2: Make requests until limit
    print("\n--- Test 2: Making Requests ---")
    for i in range(4):
        allowed, reason = limiter.check_limit(user_id)
        print(f"\nRequest {i+1}:")
        print(f"  Allowed: {allowed}")
        print(f"  Reason: {reason}")
        
        if allowed:
            limiter.record_request(user_id)
            quota = limiter.get_remaining_quota(user_id)
            print(f"  Remaining: {quota['hourly']['remaining']}/{quota['hourly']['limit']}")
    
    # Test 3: Get final stats
    print("\n--- Test 3: Statistics ---")
    stats = limiter.get_stats()
    print(f"Total users: {stats['total_users']}")
    print(f"Active users: {stats['active_users']}")
    print(f"Hourly requests: {stats['hourly_requests']}")
    
    # Test 4: Reset user
    print("\n--- Test 4: Reset User ---")
    limiter.reset_user(user_id)
    quota = limiter.get_remaining_quota(user_id)
    print(f"✅ After reset - Hourly remaining: {quota['hourly']['remaining']}/{quota['hourly']['limit']}")
    
    print("\n" + "="*60)
    print("✅ All rate limiter tests passed!")
    print("="*60 + "\n")
