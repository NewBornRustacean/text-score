//! This is an implementation for metrics to be used in various ML/DL fields.
//! for now, split_whitespace based rouge-n score is provided.
//!
use std::collections::HashMap;
use std::cmp::{min, max};

use approx::assert_abs_diff_eq;

/// Represents precision, recall, and F1 score.
///
/// The `Score` struct contains three floating-point fields: `precision`, `recall`, and `f1`.
/// These fields represent evaluation metrics commonly used in natural language processing and information retrieval.
///
/// ### Examples
/// ```
/// use metrics_rs::Score;
///
/// // Create a Score instance
/// let score = Score {
///     precision: 0.8,
///     recall: 0.7,
///     f1: 0.75,
/// };
///
/// // Access individual fields
/// assert_eq!(score.precision, 0.8);
/// assert_eq!(score.recall, 0.7);
/// assert_eq!(score.f1, 0.75);
/// ```
pub struct Score{
    pub precision: f32,
    pub recall:f32,
    pub f1:f32,
}
fn precision(true_pos:u32, false_pos:u32) -> f32{
    return true_pos as f32 /((true_pos+false_pos) as f32);
}
fn recall(true_pos:u32, false_neg:u32) -> f32{
    return true_pos as f32 /((true_pos+false_neg) as f32);
}
fn f1(precision: f32, recall: f32) -> f32{
    return 2.0*(precision*recall)/(precision+recall);
}

/// Creates n-grams from a list of tokens.
///
/// Given a list of tokens and the desired size `n`, this function generates n-grams,
/// which are contiguous sequences of `n` tokens from the input list.
///
/// ### Arguments
///
/// * `tokens` - A vector of string slices representing individual tokens.
/// * `n` - The size of the n-grams to be created.
///
/// ### Returns
///
/// A `HashMap` where keys are n-grams (represented as vectors of string slices) and values
/// are the counts of each n-gram in the input sequence.
///
/// ### Examples
///
/// ```
/// use std::collections::HashMap;
/// use metrics_rs::create_ngrams;
///
/// let tokens = vec!["this", "is", "an", "example"];
/// let n = 2;
///
/// let ngrams = create_ngrams(tokens, n);
///
/// // The result may look like: {"this is": 1, "is an": 1, "an example": 1}
/// ```
///
/// ### Note
///
/// - The function uses a sliding window approach to iterate through the input `tokens`
///   and create n-grams of the specified size `n`.
/// - The resulting n-grams are stored in a `HashMap`, where each key is an n-gram,
///   and the corresponding value is the count of occurrences of that n-gram in the input sequence.
pub fn create_ngrams(tokens: Vec<&str>, n: usize) -> HashMap<Vec<&str>, u32> {
    let mut ngrams: HashMap<Vec<&str>, u32> = HashMap::new();

    for i in 0..(tokens.len() - n + 1) {
        let ngram: Vec<&str> = tokens[i..i + n].to_vec();
        *ngrams.entry(ngram).or_insert(0) += 1;
    }
    return ngrams;
}

/// Computes precision, recall, and F1 score based on n-grams.
///
/// Given two HashMaps representing the n-grams of predicted and target sequences,
/// this function calculates precision, recall, and F1 score for the prediction.
///
/// ### Arguments
///
/// * `predicted_ngrams` - A HashMap containing n-grams and their counts for the predicted sequence.
/// * `target_ngrams` - A HashMap containing n-grams and their counts for the target (reference) sequence.
///
/// ### Returns
///
/// A `Score` struct containing precision, recall, and F1 score for the prediction based on n-grams.
///
/// ### Examples
///
/// ```
/// use std::collections::HashMap;
/// use metrics_rs::{ngram_based_score, Score}; // Replace with the actual module name
///
/// let predicted_ngrams = hashmap! { vec!["this", "is"] => 2, vec!["is", "an"] => 1 };
/// let target_ngrams = hashmap! { vec!["this", "is"] => 3, vec!["is", "an"] => 2 };
///
/// let score = ngram_based_score(predicted_ngrams, target_ngrams);
/// println!("Precision: {}", score.precision); // Accessing precision field
/// println!("Recall: {}", score.recall);       // Accessing recall field
/// println!("F1 Score: {}", score.f1);         // Accessing f1 field
/// ```
///
/// # Note
///
/// - The function iterates through the target n-grams and computes the intersection count
///   with the predicted n-grams to calculate precision, recall, and F1 score.
/// - Precision and recall are calculated using the standard formulas, and F1 score is computed
///   using the `f1` function defined in the module.
/// - The resulting scores are returned in a `Score` struct.
pub fn ngram_based_score(predicted_ngrams:HashMap<Vec<&str>, u32>, target_ngrams:HashMap<Vec<&str>, u32>) -> Score{
    let mut intersection_ngrams_count: u32=0;
    let target_ngrams_count:u32 = target_ngrams.values().map(|&v| v).sum();
    let prediction_ngrams_count:u32= predicted_ngrams.values().map(|&v| v).sum();

    for (ngram, target_cnt) in target_ngrams.iter(){
        intersection_ngrams_count += min(target_cnt, predicted_ngrams.get(ngram).unwrap_or(&0));

    }
    let p:f32 = intersection_ngrams_count as f32/ max(prediction_ngrams_count, 1) as f32;
    let r:f32 = intersection_ngrams_count as f32/ max(target_ngrams_count, 1) as f32;
    let f:f32 = f1(p, r);

    return Score{precision:p, recall:r, f1:f};
}


/// Computes ROUGE scores based on n-grams for a given input and reference text.
///
/// ROUGE (Recall-Oriented Understudy for Gisting Evaluation) is a metric commonly used
/// in natural language processing to evaluate the quality of text summaries or translations.
/// This function calculates precision, recall, and F1 score based on n-grams for the provided input
/// text and reference text.
///
/// ### Arguments
///
/// * `input` - The input text to be evaluated.
/// * `reference` - The reference text, considered as the ground truth or gold standard.
/// * `n` - The size of n-grams to be used in the evaluation.
///
/// ### Returns
///
/// A `Result` containing a `Score` struct if successful, or an error message if `n` is less than 1.
///
/// ### Examples
///
/// ```
/// use metrics_rs::{rouge, Score}; // Replace with the actual module name
///
/// let input_text = "This is a sample sentence for evaluation.";
/// let reference_text = "This is a sample sentence for testing.";
/// let n = 2;
///
/// match rouge(input_text, reference_text, n) {
///     Ok(score) => {
///         println!("Precision: {}", score.precision); // Accessing precision field
///         println!("Recall: {}", score.recall);       // Accessing recall field
///         println!("F1 Score: {}", score.f1);         // Accessing f1 field
///     }
///     Err(err) => println!("Error: {}", err),
/// }
/// ```
///
/// # Note
///
/// - The function checks if the specified `n` is greater than or equal to 1. If not, it returns an error.
/// - The input and reference texts are tokenized into words, and n-grams are created using the `create_ngrams` function.
/// - The n-gram based scores are then calculated using the `ngram_based_score` function.
/// - The resulting scores are returned in a `Score` struct if the operation is successful.
pub fn rouge(input:&str, reference: &str, n:usize) -> Result<Score, String>{
    if n<1{
        return Err("n should be 1>=1".to_string());
    }
    let input_words = input.split_whitespace().collect();
    let reference_words = reference.split_whitespace().collect();

    // create n-grams
    let mut input_ngrams = create_ngrams(input_words, n);
    let mut reference_ngrams = create_ngrams(reference_words, n);

    // get n-gram based f1 score
    return Ok(ngram_based_score(input_ngrams, reference_ngrams));
}


#[cfg(test)]
mod test_metrics  {
    use super::*;

    #[test]
    fn test_precision(){
        assert_eq!(1.0, precision(10, 0));
        assert_eq!(2.0/3.0, precision(10, 5));
    }
    #[test]
    fn test_recall(){
        assert_eq!(1.0, recall(10, 0));
        assert_eq!(2.0/3.0, recall(10, 5));

    }
    #[test]
    fn test_f1(){
        assert_eq!(1.0, f1(1.0, 1.0));
        assert_eq!(0.0, f1(0.0, 1.0));
        assert_eq!(0.0, f1(1.0, 0.0));

    }

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
        let score = rouge("this is identical case.", "this is identical case.", 1).unwrap();
        assert_eq!(1.0, score.f1);

        // subset: 4/5 correct
        let score = rouge("this is identical case.", "wow this is identical case.", 1).unwrap();
        assert_abs_diff_eq!(0.888,  score.f1, epsilon = 1e-3);

        // duplicated words case: 1.0
        let score = rouge("it is what it is.", "it is what it is.", 1).unwrap();
        assert_abs_diff_eq!(1.0,  score.f1, epsilon = 1e-3);

        // duplicated words case: 5/6 correct, p=1, r=5/6.
        let score = rouge("it is what it is.", "it is really what it is.", 1).unwrap();
        assert_abs_diff_eq!(f1(1.0, 5.0/6.0),  score.f1, epsilon = 1e-3);
    }
}