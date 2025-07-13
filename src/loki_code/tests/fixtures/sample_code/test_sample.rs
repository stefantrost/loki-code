/**
 * Sample Rust code for testing Tree-sitter analysis.
 */

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct Calculator {
    history: Vec<String>,
}

impl Calculator {
    pub fn new() -> Self {
        Calculator {
            history: Vec::new(),
        }
    }
    
    pub fn add(&mut self, a: i32, b: i32) -> i32 {
        let result = a + b;
        self.history.push(format!("{} + {} = {}", a, b, result));
        result
    }
    
    pub fn multiply(&mut self, a: i32, b: i32) -> i32 {
        let result = a * b;
        self.history.push(format!("{} * {} = {}", a, b, result));
        result
    }
    
    pub fn get_history(&self) -> &Vec<String> {
        &self.history
    }
}

pub fn fibonacci(n: u32) -> u32 {
    match n {
        0 => 0,
        1 => 1,
        _ => fibonacci(n - 1) + fibonacci(n - 2),
    }
}

#[derive(Debug)]
pub struct ProcessingOptions {
    pub verbose: bool,
    pub timeout: u32,
}

impl Default for ProcessingOptions {
    fn default() -> Self {
        ProcessingOptions {
            verbose: false,
            timeout: 30,
        }
    }
}

pub fn complex_function(
    data: HashMap<String, String>, 
    options: Option<ProcessingOptions>
) -> HashMap<String, String> {
    let opts = options.unwrap_or_default();
    let mut result = HashMap::new();
    
    result.insert("status".to_string(), "processing".to_string());
    result.insert("input_size".to_string(), data.len().to_string());
    
    // Process data
    for (key, value) in data.iter() {
        result.insert(key.clone(), value.to_uppercase());
    }
    
    result.insert("status".to_string(), "completed".to_string());
    result
}

// Generic function with traits
pub fn process_items<T: Clone + std::fmt::Debug>(items: Vec<T>) -> Vec<T> {
    items.into_iter().collect()
}

// Error handling with Result
#[derive(Debug)]
pub enum CalculationError {
    DivisionByZero,
    InvalidInput,
}

impl std::fmt::Display for CalculationError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            CalculationError::DivisionByZero => write!(f, "Division by zero"),
            CalculationError::InvalidInput => write!(f, "Invalid input"),
        }
    }
}

impl std::error::Error for CalculationError {}

pub fn safe_divide(a: f64, b: f64) -> Result<f64, CalculationError> {
    if b == 0.0 {
        Err(CalculationError::DivisionByZero)
    } else {
        Ok(a / b)
    }
}

fn main() {
    let mut calc = Calculator::new();
    println!("5 + 3 = {}", calc.add(5, 3));
    println!("4 * 7 = {}", calc.multiply(4, 7));
    
    match safe_divide(10.0, 2.0) {
        Ok(result) => println!("10 / 2 = {}", result),
        Err(e) => println!("Error: {}", e),
    }
}