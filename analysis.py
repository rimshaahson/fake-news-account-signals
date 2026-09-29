"""
Can account behavior identify who shares fake news?

Uses 1.3 million tweets from FakeNewsNet, each labelled by whether the news it
shares is fake or real, and asks whether the posting account's traits
(followers, activity, verification, profile settings) can predict the label.

The key design choice is how the data is split. The same accounts tweet many
times, so a random split puts one account's tweets in both the training and
test sets, and a flexible model can score well just by recognising accounts.
This script evaluates every model twice, with a random split and with an
account-grouped split, to show how much that inflates the results.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42
DATA = Path("data")
IMAGES = Path("images")
IMAGES.mkdir(exist_ok=True)

COUNTS = ["followers", "following", "statuses_count", "favorite_count"]
FLAGS = ["verified", "profile_background_tile", "profile_use_background_image"]
LABELS = {
    "log_followers": "Followers",
    "log_following": "Accounts followed",
    "log_statuses_count": "Total tweets posted",
    "log_favorite_count": "Likes on the tweet",
    "verified": "Verified account",
    "profile_background_tile": "Tiled profile background",
    "profile_use_background_image": "Uses a background image",
}
TEAL, GOLD, GREY = "#1E6F6A", "#B8893B", "#9AA5AE"


# ---------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------

def load(path, is_fake):
    df = pd.read_csv(path, usecols=["user_id"] + COUNTS + FLAGS)
    df["is_fake"] = is_fake
    return df

tweets = pd.concat([load(DATA / "FakeNewsNet_Fake.csv", 1),
                    load(DATA / "FakeNewsNet_Real.csv", 0)], ignore_index=True)

print(f"Tweets: {len(tweets):,}  (fake news: {tweets.is_fake.mean():.1%})")
print(f"Accounts: {tweets.user_id.nunique():,}")

# Counts are heavily skewed (a few accounts have millions of followers), so
# they are log-transformed before modelling.
X = pd.concat([np.log1p(tweets[COUNTS]).add_prefix("log_"),
               tweets[FLAGS].astype(int)], axis=1)
y = tweets.is_fake.to_numpy()
accounts = tweets.user_id.to_numpy()


# ---------------------------------------------------------------------
# 2. Two ways to split the data
# ---------------------------------------------------------------------

rows = np.arange(len(tweets))
random_split = train_test_split(rows, test_size=0.3, random_state=SEED, stratify=y)
grouped_split = next(GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=SEED)
                     .split(X, y, groups=accounts))

shared = len(set(accounts[random_split[0]]) & set(accounts[random_split[1]]))
print(f"Accounts in both train and test with a random split: {shared:,}")


# ---------------------------------------------------------------------
# 3. Train and evaluate
# ---------------------------------------------------------------------

def scores(actual, prob):
    pred = prob > 0.5
    return {"accuracy": accuracy_score(actual, pred),
            "auc": roc_auc_score(actual, prob),
            "recall_fake": recall_score(actual, pred),
            "precision_fake": precision_score(actual, pred)}

results, fitted = [], {}
for split_name, (train, test) in [("Random split", random_split),
                                  ("Account-grouped split", grouped_split)]:
    scaler = StandardScaler().fit(X.iloc[train])
    X_train, X_test = scaler.transform(X.iloc[train]), scaler.transform(X.iloc[test])

    logit = LogisticRegression(max_iter=1000).fit(X_train, y[train])
    results.append({"split": split_name, "model": "Logistic regression",
                    **scores(y[test], logit.predict_proba(X_test)[:, 1])})

    boost = HistGradientBoostingClassifier(max_iter=200, random_state=SEED).fit(X_train, y[train])
    results.append({"split": split_name, "model": "Gradient boosting",
                    **scores(y[test], boost.predict_proba(X_test)[:, 1])})

    fitted[split_name] = logit

results = pd.DataFrame(results).round(3)
baseline = max(y[grouped_split[1]].mean(), 1 - y[grouped_split[1]].mean())
print(f"\nBaseline (always guess the larger class): {baseline:.1%}")
print(results.to_string(index=False))
results.to_csv(IMAGES / "model_results.csv", index=False)


# ---------------------------------------------------------------------
# 4. Chart: how much a random split inflates each model
# ---------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7.5, 4.2))
models = ["Logistic regression", "Gradient boosting"]
width = 0.36
for i, (split_name, colour) in enumerate([("Random split", GREY),
                                           ("Account-grouped split", TEAL)]):
    vals = [results.query("split == @split_name and model == @m").accuracy.item() for m in models]
    bars = ax.bar(np.arange(2) + (i - 0.5) * width, vals, width, color=colour, label=split_name)
    ax.bar_label(bars, labels=[f"{v:.0%}" for v in vals], padding=3, fontsize=10)
ax.axhline(baseline, color=GOLD, linestyle="--", linewidth=1.2)
ax.text(0.5, baseline + 0.01, "Guessing: 50%", color=GOLD, ha="center", fontsize=9)
ax.set_xticks(np.arange(2), models)
ax.set_ylim(0.4, 0.85)
ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
ax.set_ylabel("Test accuracy")
ax.set_title("A random split makes gradient boosting look far better than it is",
             loc="left", fontsize=12, fontweight="bold")
ax.legend(frameon=False, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(IMAGES / "split_comparison.png", dpi=150)


# ---------------------------------------------------------------------
# 5. Chart: which account traits matter (logistic regression, grouped split)
# ---------------------------------------------------------------------

logit = fitted["Account-grouped split"]
odds = (pd.Series(np.exp(logit.coef_[0]), index=X.columns)
        .rename(index=LABELS).sort_values())

fig, ax = plt.subplots(figsize=(7.5, 4.2))
colours = [TEAL if v > 1 else GOLD for v in odds]
ax.barh(odds.index, odds - 1, left=1, color=colours, height=0.6)
ax.axvline(1, color="#52606D", linewidth=1)
for label, v in odds.items():
    ax.text(v + (0.03 if v > 1 else -0.03), label, f"{v:.2f}", va="center",
            ha="left" if v > 1 else "right", fontsize=9)
ax.set_xlim(0.3, 2.1)
ax.set_xlabel("Odds ratio per standard deviation (above 1 = more likely to share fake news)")
ax.set_title("What distinguishes accounts that share fake news", loc="left",
             fontsize=12, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(IMAGES / "odds_ratios.png", dpi=150)

print("\nOdds ratios per standard deviation (logistic regression, grouped split):")
print(odds.round(3).to_string())
