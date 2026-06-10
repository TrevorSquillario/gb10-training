---
name: media-scout
description: Searches for newly released or upcoming TV series and movies across streaming platforms, providing brief synopses, release dates, and popularity ratings.
version: 1.0.0
---

Search the web using `web_search time_range="week"` for the newly released or upcoming tv series or movies across the streaming platforms. (e.g., Netflix, Hulu, Amazon Prime, Disney+, HBO Max, Apple TV+, Peacock, Paramount+). Use delegate_task to navigate to these sites, in parallel, to extract the relevant information about the titles, including their synopses, release dates, and popularity ratings (if available). Include a `media_type` property to designate if it's a `series` (tv series, limited series, docuseries) or `movie` (movies, documentaries) Focus on titles that have generated significant buzz or anticipation among audiences and critics. 

Use hindsight_reflect using bank_id="jellyfin" to provide a generalized global summary of recently watched items

Compare the recently watched items with the newly released or upcoming titles. Highlight similarities in genres, themes, or cast members. Make a list of your top 5 recommendations based on the recently watched items and the new releases. 

Lookup each of the top recommendations using the ARR `lookup_movie` or `lookup_series`. Select the best match based on the title and year. Then, check the monitored property to see if it's being monitored in Sonarr or Radarr. 

Determine what isn't monitored, then call arr `add_movie tmdb_id=<tmdb_id>` or `add_series tvdb_id=<tvdb_id>` depending on the `media type` to add it to the respective library. Set the following properties when adding: `monitor: bool = False, search_for_movie: bool = False` or `search_for_series: bool = False`.

Provide a summary of the actions taken:

Include:
- Last/future airing date and current status (e.g. ongoing, ended, upcoming).
- Primary streaming platform
- Notable cast members or directors
- Source url from the original search results for each title.
- Titles added to Sonarr/Radarr, if any, and their monitoring status.

