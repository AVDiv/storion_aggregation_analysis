# Problem

The content data collected by the **collection engine** is not clean at some cases. The content contains text which was on the website/popups, but not related to the main content. This unwanted text needs to be removed from the content. Also, at some cases the content is not scraped, but unwanted content might be scraped. At such cases, the content needs to be removed and marked as missing before going through the aggregation process.

# Solution

Luckily, as the collection engine depends on RSS feeds, the title is extracted properly (As all the RSS feeds have a title).

The initial idea to start off with, is to use the title, and progress through the content as sentences. After a related sentence is found, the sentence will be used to find the next related sentence. This will be done until the end of the content. The related sentences will be used to form the final content.

I don't know what to name this, maybe something like "Progressive Sentence Chain Content Filtering..." or something. That for later.

# Notes for development

- The content may include HTML/escape characters, might need to consider cleaning the content before bringing to this stage.
- Need to set a mechanism to mark the article as content_invalid if the article content (except for the unwanted text) is not found.
