#!/usr/bin/env python3
"""Generate movie ratings.

A simple script to generate a basic preference profile in order to generate
personalized recommendations.

Inspired from
<https://github.com/databricks/spark-training/blob/master/machine-learning/bin/rateMovies>
"""
import time
import json


MOVIES = (
    (1, "Toy Story (1995)", "Adventure, Animation, Children, Comedy, Fantasy"),
    (780, "Independence Day (1996)", "Action, Adventure, Sci-Fi, Thriller"),
    (590, "Dances with Wolves (1990)", "Adventure, Drama, Western"),
    (1210, "Star Wars: Episode VI - Return of the Jedi (1983)", "Action, Adventure, Sci-Fi"),
    (648, "Mission: Impossible (1996)", "Action, Adventure, Mystery, Thriller"),
    (344, "Ace Ventura: Pet Detective (1994)", "Comedy"),
    (165, "Die Hard: With a Vengeance (1995)", "Action, Crime, Thriller"),
    (153, "Batman Forever (1995)", "Action, Adventure, Comedy, Crime"),
    (597, "Pretty Woman (1990)", "Comedy, Romance"),
    (1580, "Men in Black (1997)", "Action, Comedy, Sci-Fi"),
    (231, "Dumb & Dumber (1994)", "Adventure, Comedy"),
)

PATH = "my-ratings.txt"
UID = 138494


def main():
    print("Please rate the following movies.")
    print("1 (worst) -- 5 (best), or 0 if not seen")
    ratings = list()
    for mid, title, genres in MOVIES:
        valid = False
        while not valid:
            print("\n{}\n{}".format(title, genres))
            rating = input("your rating: ")
            try:
                r = int(rating)
                if 1 <= r <=5:
                    ratings.append({
                        "movieId": mid,
                        "userId": UID,
                        "timestamp": int(time.time()),
                        "rating": r})
                    valid = True
                elif r == 0:
                    valid = True
            except ValueError:
                print("input is invalid, try again.")
    with open(PATH, "w") as f:
        for rating in ratings:
            print(json.dumps(rating), file=f)
    print()
    print("Successfully saved {} ratings in `{}`".format(len(ratings), PATH))
    print("Your user ID: {}".format(UID))


if __name__ == "__main__":
    main()
