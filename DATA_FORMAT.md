# Sample Data Structure

This document shows the expected format for your data files.

## Music Info.csv

| track_id | track_name | artist_name | preview_url | energy | valence | tempo | instrumentalness | tags_text |
|----------|------------|-------------|-------------|--------|---------|-------|------------------|-----------|
| 1 | Bohemian Rhapsody | Queen | https://p.scdn.co/... | 0.612 | 0.279 | 144.017 | 0.002 | rock, classic rock, british |
| 2 | Stairway to Heaven | Led Zeppelin | https://p.scdn.co/... | 0.768 | 0.441 | 82.009 | 0.007 | rock, classic rock, 70s |

## User Listening History.csv

| user_id | track_id | playcount |
|---------|----------|-----------|
| user_001 | 1 | 25 |
| user_001 | 2 | 18 |
| user_002 | 1 | 42 |

Place these files in the `data/` directory and run `python train.py` to train your models.