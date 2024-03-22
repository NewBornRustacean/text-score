use metrics_rs::commons::{f1, precision, recall};

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