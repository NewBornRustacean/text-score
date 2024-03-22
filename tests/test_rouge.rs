use approx::assert_abs_diff_eq;
use metrics_rs::rouge::{create_ngrams, rouge_n};
use metrics_rs::commons::{f1, precision, recall};

#[test]
fn test_create_ngram(){
    let tokens = "I want to build awesome rust codes".split_whitespace().collect();
    let n = 2;

    let mut ngrams = create_ngrams(tokens, n);

    for (key, value) in ngrams.iter() {
        assert_eq!(ngrams.get(key).unwrap(), value);
    }
}
#[test]
fn test_rouge1() {
    // identical: 1.0
    let score = rouge_n("this is identical case.", "this is identical case.", 1).unwrap();
    assert_eq!(1.0, score.f1);

    // subset: 4/5 correct
    let score = rouge_n("this is identical case.", "wow this is identical case.", 1).unwrap();
    assert_abs_diff_eq!(0.888,  score.f1, epsilon = 1e-3);

    // duplicated words case: 1.0
    let score = rouge_n("it is what it is.", "it is what it is.", 1).unwrap();
    assert_abs_diff_eq!(1.0,  score.f1, epsilon = 1e-3);

    // duplicated words case: 5/6 correct, p=1, r=5/6.
    let score = rouge_n("it is what it is.", "it is really what it is.", 1).unwrap();
    assert_abs_diff_eq!(f1(1.0, 5.0/6.0),  score.f1, epsilon = 1e-3);
}