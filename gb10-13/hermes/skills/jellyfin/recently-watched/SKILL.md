---
name: recently-watched
description: Retrieves and summarizes recently watched items for each Jellyfin user, highlighting genres, tags, studios, and notable people.
version: 1.0.0
---

Fet a list of all the jellyfin users unless a specific username is provided.

Then, use delegate_task to spawn individual tasks for each user that calls jellyfin get_recently_watched using the user ids obtained from the previous step.

  In each task, summarize following:

    Titles: What are the titles of the recently watched items?
    Genres: What are the main genres here, and which one pops up the most?
    Tags: What kind of themes, vibes, or keywords are trending in this list?
    Studios: Which studios are involved? Are there any recurring ones?
    Popular/Notable People: Highlight some of the main or well-known names listed in the cast/people sections.

If instructed to save the memory, use the hindsight_retain bank_id="jellyfin". Provide confirmation on the success or failure of the memory retention, along with any relevant details or error messages.
