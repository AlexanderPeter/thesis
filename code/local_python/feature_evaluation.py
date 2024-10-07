from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier


def calculate_scores(
    targets,
    predictions,
    entry,
):
    accuracy = accuracy_score(targets, predictions)
    balanced_accuracy = balanced_accuracy_score(targets, predictions)
    f1_weighted = f1_score(targets, predictions, average="weighted")
    f1_micro = f1_score(targets, predictions, average="micro")
    f1_macro = f1_score(targets, predictions, average="macro")

    entry["accuracy"] = accuracy
    entry["balanced_accuracy"] = balanced_accuracy
    entry["f1_weighted"] = f1_weighted
    entry["f1_micro"] = f1_micro
    entry["f1_macro"] = f1_macro
    return entry


def evaluate_with_lr(
    features,
    targets,
    entry,
    seed=None,
    max_iter=10_000,
):
    (train_features, _, test_features) = features
    (train_targets, _, test_targets) = targets

    model_lr = LogisticRegression(max_iter=max_iter, random_state=seed)
    model_lr.fit(train_features, train_targets)
    test_pred = model_lr.predict(test_features)

    entry["n_iter"] = model_lr.n_iter_.tolist()[0]
    return calculate_scores(
        test_targets,
        test_pred,
        entry,
    )


def evaluate_with_knn(
    features,
    targets,
    entry,
    number_of_samples,
    knn_metric="cosine",
    max_neighbors=200,
    best_k_f1_score_average="macro",
):
    (train_features, valid_features, test_features) = features
    (train_targets, valid_targets, test_targets) = targets

    scores = {}
    neighbors_limit = max_neighbors
    if number_of_samples is not None:
        neighbors_limit = min(neighbors_limit, 1 + number_of_samples)

    for k in range(1, neighbors_limit):
        model_knn = KNeighborsClassifier(n_neighbors=k, metric=knn_metric)
        model_knn.fit(train_features, train_targets)
        valid_pred = model_knn.predict(valid_features)
        score = f1_score(valid_targets, valid_pred, average=best_k_f1_score_average)
        scores[k] = score

    best_k = max(scores, key=scores.get)
    model_knn = KNeighborsClassifier(n_neighbors=best_k)
    model_knn.fit(train_features, train_targets)
    test_pred = model_knn.predict(test_features)

    entry["best_k"] = best_k
    return calculate_scores(
        test_targets,
        test_pred,
        entry,
    )
