
fn precision(true_pos:u32, false_pos:u32) -> f32{
    return true_pos as f32 /((true_pos+false_pos) as f32);
}
fn recall(true_pos:u32, false_neg:u32) -> f32{
    return true_pos as f32 /((true_pos+false_neg) as f32);
}
fn f1(precision: f32, recall: f32) -> f32{
    return 2.0*(precision*recall)/(precision+recall);
}
pub fn rouge1(input:&str, reference: &str) -> f32{
    let input_words = input.split(" ");
    let reference_words = reference.split(" ");


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
    fn test_rouge1() {
        // identical: 1.0
        assert_eq!(1.0, rouge1("this is identical case.", "this is identical case."));

    }
}