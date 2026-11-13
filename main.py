import argparse
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, learning_curve, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import make_scorer

from dataloader import load_data
from result_writer import save_sample_results, save_summary


def build_model(X, max_depth=None):
    categorical = X.select_dtypes(include="object").columns
    numeric = X.columns.difference(categorical)
    preprocess = ColumnTransformer(
        [
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("numeric", "passthrough", numeric),
        ]
    )
    return Pipeline(
        [
            ("preprocess", preprocess),
            ("tree", DecisionTreeClassifier(
                random_state=42, max_depth=max_depth, class_weight="balanced"
            )),
        ]
    )


def main(subject, max_depth):
    X, y = load_data(subject)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model = build_model(X_train, max_depth)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, pos_label=0, zero_division=0),
        "recall": make_scorer(recall_score, pos_label=0, zero_division=0),
        "f1": make_scorer(f1_score, pos_label=0, zero_division=0),
        "specificity": make_scorer(recall_score, pos_label=1, zero_division=0),
    }
    scores = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring)
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    fail_probabilities = 1 - probabilities
    y_test_fail = 1 - y_test
    cv_scores = {metric: scores[f"test_{metric}"] for metric in scoring}
    test_scores = {
        "accuracy": accuracy_score(y_test, prediction),
        "precision": precision_score(y_test, prediction, pos_label=0, zero_division=0),
        "recall": recall_score(y_test, prediction, pos_label=0, zero_division=0),
        "f1": f1_score(y_test, prediction, pos_label=0, zero_division=0),
        "specificity": recall_score(y_test, prediction, pos_label=1, zero_division=0),
    }
    report = classification_report(
        y_test,
        prediction,
        target_names=["Fail", "Pass"],
        output_dict=True,
        zero_division=0,
    )
    tree = model.named_steps["tree"]
    train_prediction = model.predict(X_train)
    train_f1 = f1_score(y_train, train_prediction, pos_label=0, zero_division=0)
    test_f1 = test_scores["f1"]
    extra_scores = {
        "test probability": {
            "roc_auc": roc_auc_score(y_test_fail, fail_probabilities),
            "pr_auc": average_precision_score(y_test_fail, fail_probabilities),
        },
        "classification report": {
            "fail_precision": report["Fail"]["precision"],
            "fail_recall": report["Fail"]["recall"],
            "fail_f1": report["Fail"]["f1-score"],
            "pass_precision": report["Pass"]["precision"],
            "pass_recall": report["Pass"]["recall"],
            "pass_f1": report["Pass"]["f1-score"],
        },
        "decision tree": {
            "n_leaves": tree.get_n_leaves(),
            "max_depth": tree.get_depth(),
            "train_f1": train_f1,
            "test_f1": test_f1,
            "train_test_f1_gap": train_f1 - test_f1,
        },
    }

    print(f"Dataset: {subject} | samples: {len(X)} | pass rate: {y.mean():.1%}")
    print(
        f"5-fold CV: accuracy={cv_scores['accuracy'].mean():.3f} "
        f"+/- {cv_scores['accuracy'].std():.3f}, "
        f"F1(Fail)={cv_scores['f1'].mean():.3f} +/- {cv_scores['f1'].std():.3f}"
    )
    print(
        f"Test: accuracy={test_scores['accuracy']:.3f}, "
        f"F1(Fail)={test_scores['f1']:.3f}, "
        f"ROC-AUC={extra_scores['test probability']['roc_auc']:.3f}, "
        f"PR-AUC={extra_scores['test probability']['pr_auc']:.3f}"
    )

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    summary_path = save_summary(subject, cv_scores, test_scores, output_dir, extra_scores)
    samples_path = save_sample_results(subject, X_test, y_test, prediction, probabilities, output_dir)
    print(f"CSV summary: {summary_path}")
    print(f"CSV samples: {samples_path}")

    feature_names = model.named_steps["preprocess"].get_feature_names_out()
    importance = pd.Series(model.named_steps["tree"].feature_importances_, index=feature_names).sort_values(ascending=False).head(10).sort_values()

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    ConfusionMatrixDisplay.from_predictions(y_test, prediction, display_labels=["Fail", "Pass"], ax=axes[0, 0], cmap="Blues")
    importance.plot.barh(ax=axes[0, 1], color="steelblue")
    axes[0, 1].set_title("Top 10 feature importance")
    axes[0, 1].set_xlabel("Importance")
    fpr, tpr, _ = roc_curve(y_test_fail, fail_probabilities)
    axes[1, 0].plot(fpr, tpr, label=f"AUC={extra_scores['test probability']['roc_auc']:.3f}")
    axes[1, 0].plot([0, 1], [0, 1], "--", color="gray")
    axes[1, 0].set(title="ROC Curve", xlabel="False positive rate", ylabel="Recall")
    axes[1, 0].legend()
    precision, recall, _ = precision_recall_curve(y_test_fail, fail_probabilities)
    axes[1, 1].plot(recall, precision, label=f"AP={extra_scores['test probability']['pr_auc']:.3f}")
    axes[1, 1].set(title="Precision-Recall Curve", xlabel="Recall", ylabel="Precision")
    axes[1, 1].legend()
    fig.suptitle(f"Decision Tree - {subject}")
    fig.tight_layout()
    figure_path = output_dir / f"{subject}_results.png"
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)
    print(f"Figure: {figure_path}")

    tree_path = output_dir / f"{subject}_tree_structure.png"
    tree_figure = plt.figure(figsize=(max(18, tree.get_n_leaves() * 0.35), max(10, tree.get_depth() * 0.7)))
    plot_tree(
        tree,
        feature_names=feature_names,
        class_names=["Fail", "Pass"],
        filled=True,
        rounded=True,
        proportion=True,
        precision=2,
        # fontsize=21,
    )
    tree_figure.tight_layout()
    tree_figure.savefig(tree_path, dpi=150, bbox_inches="tight")
    plt.close(tree_figure)
    print(f"Tree structure: {tree_path}")

    sizes, train_scores, validation_scores = learning_curve(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=make_scorer(f1_score, pos_label=0, zero_division=0),
        train_sizes=[0.2, 0.4, 0.6, 0.8, 1.0],
    )
    learning_path = output_dir / f"{subject}_learning_curve.png"
    plt.figure(figsize=(7, 5))
    plt.plot(sizes, train_scores.mean(axis=1), "o-", label="Train F1")
    plt.plot(sizes, validation_scores.mean(axis=1), "o-", label="CV F1")
    plt.xlabel("Training samples")
    plt.ylabel("F1 (Fail)")
    plt.title(f"Learning Curve - {subject}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(learning_path, dpi=150)
    plt.close()
    print(f"Learning curve: {learning_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Student performance classification")
    parser.add_argument("--subject", choices=["mat", "por", "both"], default="both")
    parser.add_argument("--max-depth", type=int, default=None)
    args = parser.parse_args()
    main(args.subject, args.max_depth)
