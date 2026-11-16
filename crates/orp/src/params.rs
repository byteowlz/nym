use ort::ep::ExecutionProviderDispatch;


/// Represents the set of parameters for the inference engine
/// 
/// The easiest way to instanciate sound parameters is to use the
/// `default()` constructor and then use individual setters as needed.
pub struct RuntimeParameters {
    /// Number ot threads (default: 4)
    threads: usize,
    /// Execution providers (default: none (-> CPU))
    execution_providers: Vec<ExecutionProviderDispatch>,
}

impl RuntimeParameters {
    pub fn new(threads: usize, execution_providers: impl IntoIterator<Item = ExecutionProviderDispatch>) -> Self {
        Self {
            threads,
            execution_providers: execution_providers.into_iter().collect(),
        }
    }

    /// Set the number ot threads (default: 4)
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.threads = threads;
        self
    }

    /// Set the execution providers (default: none, ie. CPU)
    pub fn with_execution_providers(mut self, execution_providers: impl IntoIterator<Item = ExecutionProviderDispatch>) -> Self {
        self.execution_providers = execution_providers.into_iter().collect();
        self
    }

    /// Get the number of threads
    pub fn threads(&self) -> usize {
        self.threads
    }

    /// Get the execution providers
    pub fn execution_providers(&self) -> &[ExecutionProviderDispatch] {
        &self.execution_providers
    }
    
    // Move out the execution providers
    pub(crate) fn into_execution_providers(self) -> std::vec::IntoIter<ExecutionProviderDispatch> {
        self.execution_providers.into_iter()
    }
        
}


impl Default for RuntimeParameters {
    fn default() -> Self {
        Self::new(4, [])
    }
}