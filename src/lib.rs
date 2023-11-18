use std::collections::HashMap;
use std::cmp::{min, max};

pub struct Score{
    precision: f32,
    recall:f32,
    f1:f32,
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

fn create_ngrams(tokens: Vec<&str>, n: usize) -> HashMap<Vec<&str>, u32> {
    let mut ngrams: HashMap<Vec<&str>, u32> = HashMap::new();

    for i in 0..(tokens.len() - n + 1) {
        let ngram: Vec<&str> = tokens[i..i + n].to_vec();
        *ngrams.entry(ngram).or_insert(0) += 1;
    }
    return ngrams;
}

fn ngram_based_score(predicted_ngrams:HashMap<Vec<&str>, u32>, target_ngrams:HashMap<Vec<&str>, u32>) -> Score{
    let mut intersection_ngrams_count: u32=0;
    let target_ngrams_count:u32 = target_ngrams.values().map(|&v| v).sum();
    let prediction_ngrams_count:u32= predicted_ngrams.values().map(|&v| v).sum();

    for (ngram, target_cnt) in target_ngrams.iter(){
        intersection_ngrams_count += min(target_cnt, predicted_ngrams.get(ngram).unwrap_or(&0));

    }

    let p:f32 = (intersection_ngrams_count / max(prediction_ngrams_count, 1)) as f32;
    let r:f32 = (intersection_ngrams_count / max(target_ngrams_count, 1)) as f32;
    let f:f32 = f1(p, r);

    return Score{precision:p, recall:r, f1:f};
}

pub fn rouge(input:&str, reference: &str, n:usize) -> Result<Score, String>{
    if n<1{
        return Err("n should be 1>=1".to_string());
    }
    else{
        let input_words = input.split_whitespace().collect();
        let reference_words = reference.split_whitespace().collect();

        // create n-grams
        let mut input_ngrams = create_ngrams(input_words, n);
        let mut reference_ngrams = create_ngrams(reference_words, n);

        // get n-gram based f1 score
        return Ok(ngram_based_score(input_ngrams, reference_ngrams));
    }
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
        assert_eq!(1.0, f1(1.0, 1.0));

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

        // // subset:
        // let score = rouge("this is identical case.", "wow this is identical case.", 1).unwrap();
        // assert_eq!(1.0, score.f1);
    }
}