"""Implementation of TwitterAPI."""


class TwitterAPI:
    """A Twitter API for BFCL evaluation."""

    def __init__(self):
        self.username = ""
        self.password = ""
        self.authenticated = False
        self.tweets = {}
        self.comments = {}
        self.retweets = {}
        self.following_list = []
        self.tweet_counter = 0

    def _load_scenario(self, config):
        """Load the Twitter API state from configuration."""
        for key, value in config.items():
            setattr(self, key, value)

    def login(self, username, password):
        """Log in to Twitter."""
        if username == self.username and password == self.password:
            self.authenticated = True
            return {"status": "success", "message": f"Logged in as {username}"}
        else:
            return {"status": "error", "message": "Invalid username or password"}

    def logout(self):
        """Log out from Twitter."""
        if self.authenticated:
            self.authenticated = False
            return {"status": "success", "message": "Logged out successfully"}
        else:
            return {"status": "error", "message": "Not logged in"}

    def post_tweet(self, content, tags=None, mentions=None):
        """Post a new tweet."""
        if not self.authenticated:
            return {"status": "error", "message": "Not authenticated"}

        if not content:
            return {"status": "error", "message": "Tweet content cannot be empty"}

        tweet_id = self.tweet_counter
        self.tweet_counter += 1

        self.tweets[str(tweet_id)] = {
            "id": tweet_id,
            "content": content,
            "username": self.username,
            "tags": tags or [],
            "mentions": mentions or [],
        }

        return {
            "status": "success",
            "message": "Tweet posted successfully",
            "tweet_id": tweet_id,
        }

    def get_tweets(self, username=None):
        """Get tweets by a specific user or all tweets if username is None."""
        tweets_to_return = {}

        for tweet_id, tweet in self.tweets.items():
            if username is None or tweet["username"] == username:
                tweets_to_return[tweet_id] = tweet

        return tweets_to_return

    def search_tweets(self, query):
        """Search tweets by content."""
        results = {}

        for tweet_id, tweet in self.tweets.items():
            if query.lower() in tweet["content"].lower():
                results[tweet_id] = tweet

        return results

    def follow_user(self, username):
        """Follow a user."""
        if not self.authenticated:
            return {"status": "error", "message": "Not authenticated"}

        if username == self.username:
            return {"status": "error", "message": "Cannot follow yourself"}

        if username in self.following_list:
            return {"status": "error", "message": f"Already following {username}"}

        self.following_list.append(username)

        return {"status": "success", "message": f"Now following {username}"}

    def unfollow_user(self, username):
        """Unfollow a user."""
        if not self.authenticated:
            return {"status": "error", "message": "Not authenticated"}

        if username not in self.following_list:
            return {"status": "error", "message": f"Not following {username}"}

        self.following_list.remove(username)

        return {"status": "success", "message": f"Unfollowed {username}"}

    def get_following(self):
        """Get the list of users being followed."""
        if not self.authenticated:
            return {"status": "error", "message": "Not authenticated"}

        return {"status": "success", "following": self.following_list}

    def comment_on_tweet(self, tweet_id, content):
        """Comment on a tweet."""
        if not self.authenticated:
            return {"status": "error", "message": "Not authenticated"}

        tweet_id_str = str(tweet_id)
        if tweet_id_str not in self.tweets:
            return {"status": "error", "message": f"Tweet {tweet_id} not found"}

        if tweet_id_str not in self.comments:
            self.comments[tweet_id_str] = []

        comment_id = len(self.comments[tweet_id_str])
        comment = {"id": comment_id, "content": content, "username": self.username}

        self.comments[tweet_id_str].append(comment)

        return {
            "status": "success",
            "message": "Comment added successfully",
            "comment_id": comment_id,
        }

    def retweet(self, tweet_id):
        """Retweet a tweet."""
        if not self.authenticated:
            return {"status": "error", "message": "Not authenticated"}

        tweet_id_str = str(tweet_id)
        if tweet_id_str not in self.tweets:
            return {"status": "error", "message": f"Tweet {tweet_id} not found"}

        if self.username not in self.retweets:
            self.retweets[self.username] = []

        if tweet_id_str in self.retweets[self.username]:
            return {"status": "error", "message": f"Already retweeted tweet {tweet_id}"}

        self.retweets[self.username].append(tweet_id_str)

        return {"status": "success", "message": f"Retweeted tweet {tweet_id}"}
