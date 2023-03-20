# Book Recommender

The Book Recommender is a simple book recommendation algorithm, using the [goodbooks-10k](https://www.kaggle.com/datasets/zygmunt/goodbooks-10k?select=books.csv) dataset. The dataset contains 10'000 popular books, as well as a list of tags these books have been tagged with by readers. The algorithm uses cosine similarity to recommend similar books based on tag data.

## Quickstart
recommender.py will first read the data contained in the /data directory, preprocess the data and then ask the user for input. To receive recommendations for a book, the **book id** must be entered. Book ids are given as the second column in the books.csv file and must currently be looked up manually. For example, line 4 of books.csv contains information for the book "Twilight" by Stephenie Meyer. The second column shows that the corresponding book id is 41865. Entering this id yields the following output:

```
Please enter a valid book id for which you would like to receive recommendations. Enter 'x' to quit.
41865
Recommendations for Twilight (Twilight, #1) by Stephenie Meyer:
 +++ Miss Peregrine’s Home for Peculiar Children (Miss Peregrine’s Peculiar Children, #1) by Ransom Riggs
 +++ Eclipse (Twilight, #3) by Stephenie Meyer
 +++ The Short Second Life of Bree Tanner: An Eclipse Novella (Twilight, #3.5) by Stephenie Meyer
 +++ Spells (Wings, #2) by Aprilynne Pike
 +++ Shadow Kiss (Vampire Academy, #3) by Richelle Mead
```

## Notes
Preprocessing the book tags may require some time. To speed up the process, the number of tags to consider can be altered by editing the tag_num argument when initializing a Recommender object. Lower tag counts result in faster preprocessing, but may decrease recommendation accuracy.
