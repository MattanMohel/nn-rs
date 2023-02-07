use crate::matrix::Mat;

/// A wrapper for a dataset
pub trait Dataset {
    /// Initializes the dataset
    fn load_dataset() -> Self;
    /// Returns the train set sample [Mat]ounts
    fn train_len(&self) -> usize;
    /// Returns the test set sample [Mat]ounts
    fn test_len(&self) -> usize;
    /// Returns `(train_data, train_labels)`
    fn train_set(&self) -> (&[Mat], &[Mat]);
    /// Returns `(test_data, test_labels)`
    fn test_set(&self) -> (&[Mat], &[Mat]);

    /// Returns the training data
    fn train_data(&self) -> &[Mat] {
        self.train_set().0
    }

    /// Returns the training labels
    fn train_labels(&self) -> &[Mat] {
        self.train_set().1
    } 
    
    /// Returns the testing data
    fn test_data(&self) -> &[Mat] {
        self.test_set().0
    }

    /// Returns the testing labels
    fn test_labels(&self) -> &[Mat] {
        self.test_set().1
    } 
}