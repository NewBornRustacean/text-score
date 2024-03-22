
/// Represents precision, recall, and F1 score.
///
/// The `Score` struct contains three floating-point fields: `precision`, `recall`, and `f1`.
/// These fields represent evaluation metrics commonly used in natural language processing and information retrieval.
///
/// ### Examples
/// ```
/// use metrics_rs::commons::Score;
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
pub fn precision(true_pos:u32, false_pos:u32) -> f32{
    return true_pos as f32 /((true_pos+false_pos) as f32);
}
pub fn recall(true_pos:u32, false_neg:u32) -> f32{
    return true_pos as f32 /((true_pos+false_neg) as f32);
}
pub fn f1(precision: f32, recall: f32) -> f32{
    return 2.0*(precision*recall)/(precision+recall);
}