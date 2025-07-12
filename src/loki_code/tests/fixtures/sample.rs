/**
 * Sample Rust file for testing code analysis functionality.
 * 
 * This file contains various Rust constructs to test the completeness
 * of the Tree-sitter parsing and analysis capabilities.
 */

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt::{Display, Formatter, Result as FmtResult};
use std::error::Error;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

// Type aliases
type TaskId = u32;
type UserId = String;
type Priority = u8;

// Constants
const MAX_TASKS: usize = 1000;
const DEFAULT_PRIORITY: Priority = 2;
const PRIORITY_LEVELS: [&str; 4] = ["low", "medium", "high", "critical"];

// Enums with associated data
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaskStatus {
    Pending,
    InProgress { started_at: std::time::Instant },
    Completed { completed_at: std::time::Instant },
    Cancelled { reason: String },
}

#[derive(Debug, Clone, PartialEq)]
pub enum PriorityLevel {
    Low = 1,
    Medium = 2,
    High = 3,
    Critical = 4,
}

impl From<Priority> for PriorityLevel {
    fn from(priority: Priority) -> Self {
        match priority {
            1 => PriorityLevel::Low,
            2 => PriorityLevel::Medium,
            3 => PriorityLevel::High,
            4 => PriorityLevel::Critical,
            _ => PriorityLevel::Medium,
        }
    }
}

impl Display for PriorityLevel {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        let level_str = match self {
            PriorityLevel::Low => "low",
            PriorityLevel::Medium => "medium",
            PriorityLevel::High => "high",
            PriorityLevel::Critical => "critical",
        };
        write!(f, "{}", level_str)
    }
}

// Custom error types
#[derive(Debug)]
pub enum TaskManagerError {
    TaskNotFound(TaskId),
    MaxTasksReached,
    InvalidPriority(Priority),
    InvalidInput(String),
}

impl Display for TaskManagerError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            TaskManagerError::TaskNotFound(id) => write!(f, "Task with ID {} not found", id),
            TaskManagerError::MaxTasksReached => write!(f, "Maximum number of tasks ({}) reached", MAX_TASKS),
            TaskManagerError::InvalidPriority(p) => write!(f, "Invalid priority level: {}", p),
            TaskManagerError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
        }
    }
}

impl Error for TaskManagerError {}

// Traits
pub trait Identifiable {
    fn get_id(&self) -> TaskId;
}

pub trait Timestamped {
    fn created_at(&self) -> std::time::Instant;
    fn updated_at(&self) -> std::time::Instant;
}

pub trait Serializable {
    fn to_json(&self) -> serde_json::Value;
    fn from_json(json: &serde_json::Value) -> Result<Self, Box<dyn Error>>
    where
        Self: Sized;
}

// Structs with various features
#[derive(Debug, Clone)]
pub struct Task {
    id: TaskId,
    title: String,
    description: Option<String>,
    priority: PriorityLevel,
    status: TaskStatus,
    tags: HashSet<String>,
    assignee: Option<UserId>,
    created_at: std::time::Instant,
    updated_at: std::time::Instant,
    dependencies: Vec<TaskId>,
}

impl Task {
    /// Create a new task with the given title
    pub fn new(id: TaskId, title: String) -> Self {
        let now = std::time::Instant::now();
        Self {
            id,
            title,
            description: None,
            priority: PriorityLevel::Medium,
            status: TaskStatus::Pending,
            tags: HashSet::new(),
            assignee: None,
            created_at: now,
            updated_at: now,
            dependencies: Vec::new(),
        }
    }

    /// Create a new task with all options
    pub fn with_options(
        id: TaskId,
        title: String,
        description: Option<String>,
        priority: PriorityLevel,
        assignee: Option<UserId>,
        tags: Vec<String>,
    ) -> Self {
        let now = std::time::Instant::now();
        let mut task = Self {
            id,
            title,
            description,
            priority,
            status: TaskStatus::Pending,
            tags: HashSet::new(),
            assignee,
            created_at: now,
            updated_at: now,
            dependencies: Vec::new(),
        };
        
        for tag in tags {
            task.add_tag(tag);
        }
        
        task
    }

    /// Get task title
    pub fn title(&self) -> &str {
        &self.title
    }

    /// Get task description
    pub fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }

    /// Get task priority
    pub fn priority(&self) -> &PriorityLevel {
        &self.priority
    }

    /// Get task status
    pub fn status(&self) -> &TaskStatus {
        &self.status
    }

    /// Get task assignee
    pub fn assignee(&self) -> Option<&str> {
        self.assignee.as_deref()
    }

    /// Get task tags
    pub fn tags(&self) -> impl Iterator<Item = &String> {
        self.tags.iter()
    }

    /// Get task dependencies
    pub fn dependencies(&self) -> &[TaskId] {
        &self.dependencies
    }

    /// Update task title
    pub fn set_title(&mut self, title: String) -> Result<(), TaskManagerError> {
        if title.trim().is_empty() {
            return Err(TaskManagerError::InvalidInput("Title cannot be empty".to_string()));
        }
        self.title = title.trim().to_string();
        self.touch();
        Ok(())
    }

    /// Update task description
    pub fn set_description(&mut self, description: Option<String>) {
        self.description = description.map(|d| d.trim().to_string()).filter(|d| !d.is_empty());
        self.touch();
    }

    /// Update task priority
    pub fn set_priority(&mut self, priority: PriorityLevel) {
        self.priority = priority;
        self.touch();
    }

    /// Assign task to user
    pub fn assign_to(&mut self, assignee: Option<UserId>) {
        self.assignee = assignee;
        self.touch();
    }

    /// Add a tag to the task
    pub fn add_tag(&mut self, tag: String) {
        if !tag.trim().is_empty() {
            self.tags.insert(tag.trim().to_lowercase());
            self.touch();
        }
    }

    /// Remove a tag from the task
    pub fn remove_tag(&mut self, tag: &str) -> bool {
        let removed = self.tags.remove(&tag.trim().to_lowercase());
        if removed {
            self.touch();
        }
        removed
    }

    /// Check if task has a specific tag
    pub fn has_tag(&self, tag: &str) -> bool {
        self.tags.contains(&tag.trim().to_lowercase())
    }

    /// Add a dependency
    pub fn add_dependency(&mut self, task_id: TaskId) {
        if !self.dependencies.contains(&task_id) && task_id != self.id {
            self.dependencies.push(task_id);
            self.touch();
        }
    }

    /// Remove a dependency
    pub fn remove_dependency(&mut self, task_id: TaskId) -> bool {
        if let Some(pos) = self.dependencies.iter().position(|&id| id == task_id) {
            self.dependencies.remove(pos);
            self.touch();
            true
        } else {
            false
        }
    }

    /// Mark task as in progress
    pub fn start(&mut self) -> Result<(), TaskManagerError> {
        match self.status {
            TaskStatus::Pending => {
                self.status = TaskStatus::InProgress {
                    started_at: std::time::Instant::now(),
                };
                self.touch();
                Ok(())
            }
            _ => Err(TaskManagerError::InvalidInput(
                "Task can only be started if it's pending".to_string(),
            )),
        }
    }

    /// Mark task as completed
    pub fn complete(&mut self) -> Result<(), TaskManagerError> {
        match self.status {
            TaskStatus::Pending | TaskStatus::InProgress { .. } => {
                self.status = TaskStatus::Completed {
                    completed_at: std::time::Instant::now(),
                };
                self.touch();
                Ok(())
            }
            _ => Err(TaskManagerError::InvalidInput(
                "Task is already completed or cancelled".to_string(),
            )),
        }
    }

    /// Cancel task with reason
    pub fn cancel(&mut self, reason: String) -> Result<(), TaskManagerError> {
        match self.status {
            TaskStatus::Completed { .. } => Err(TaskManagerError::InvalidInput(
                "Cannot cancel a completed task".to_string(),
            )),
            _ => {
                self.status = TaskStatus::Cancelled { reason };
                self.touch();
                Ok(())
            }
        }
    }

    /// Get task duration if completed
    pub fn duration(&self) -> Option<Duration> {
        match (&self.status, &self.status) {
            (TaskStatus::Completed { completed_at }, _) => {
                Some(completed_at.duration_since(self.created_at))
            }
            _ => None,
        }
    }

    /// Check if task is completed
    pub fn is_completed(&self) -> bool {
        matches!(self.status, TaskStatus::Completed { .. })
    }

    /// Check if task is in progress
    pub fn is_in_progress(&self) -> bool {
        matches!(self.status, TaskStatus::InProgress { .. })
    }

    /// Check if task is pending
    pub fn is_pending(&self) -> bool {
        matches!(self.status, TaskStatus::Pending)
    }

    /// Check if task is cancelled
    pub fn is_cancelled(&self) -> bool {
        matches!(self.status, TaskStatus::Cancelled { .. })
    }

    /// Update the updated_at timestamp
    fn touch(&mut self) {
        self.updated_at = std::time::Instant::now();
    }

    /// Clone task with new ID
    pub fn clone_with_id(&self, new_id: TaskId) -> Self {
        let mut cloned = self.clone();
        cloned.id = new_id;
        cloned.status = TaskStatus::Pending;
        cloned.created_at = std::time::Instant::now();
        cloned.updated_at = cloned.created_at;
        cloned
    }
}

impl Identifiable for Task {
    fn get_id(&self) -> TaskId {
        self.id
    }
}

impl Timestamped for Task {
    fn created_at(&self) -> std::time::Instant {
        self.created_at
    }

    fn updated_at(&self) -> std::time::Instant {
        self.updated_at
    }
}

impl Display for Task {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(
            f,
            "Task #{}: {} [{}] - {}",
            self.id,
            self.title,
            self.priority,
            match &self.status {
                TaskStatus::Pending => "Pending".to_string(),
                TaskStatus::InProgress { .. } => "In Progress".to_string(),
                TaskStatus::Completed { .. } => "Completed".to_string(),
                TaskStatus::Cancelled { reason } => format!("Cancelled: {}", reason),
            }
        )
    }
}

// Advanced TaskManager with thread safety
#[derive(Debug)]
pub struct TaskManager {
    tasks: HashMap<TaskId, Task>,
    next_id: TaskId,
    task_count_by_priority: HashMap<PriorityLevel, usize>,
    task_count_by_status: HashMap<String, usize>,
}

impl TaskManager {
    /// Create a new task manager
    pub fn new() -> Self {
        Self {
            tasks: HashMap::new(),
            next_id: 1,
            task_count_by_priority: HashMap::new(),
            task_count_by_status: HashMap::new(),
        }
    }

    /// Create a new task
    pub fn create_task(&mut self, title: String) -> Result<TaskId, TaskManagerError> {
        if self.tasks.len() >= MAX_TASKS {
            return Err(TaskManagerError::MaxTasksReached);
        }

        if title.trim().is_empty() {
            return Err(TaskManagerError::InvalidInput("Title cannot be empty".to_string()));
        }

        let task_id = self.next_id;
        self.next_id += 1;

        let task = Task::new(task_id, title);
        self.update_statistics(&task, true);
        self.tasks.insert(task_id, task);

        Ok(task_id)
    }

    /// Create a task with full options
    pub fn create_task_with_options(
        &mut self,
        title: String,
        description: Option<String>,
        priority: PriorityLevel,
        assignee: Option<UserId>,
        tags: Vec<String>,
    ) -> Result<TaskId, TaskManagerError> {
        if self.tasks.len() >= MAX_TASKS {
            return Err(TaskManagerError::MaxTasksReached);
        }

        if title.trim().is_empty() {
            return Err(TaskManagerError::InvalidInput("Title cannot be empty".to_string()));
        }

        let task_id = self.next_id;
        self.next_id += 1;

        let task = Task::with_options(task_id, title, description, priority, assignee, tags);
        self.update_statistics(&task, true);
        self.tasks.insert(task_id, task);

        Ok(task_id)
    }

    /// Get a task by ID
    pub fn get_task(&self, task_id: TaskId) -> Option<&Task> {
        self.tasks.get(&task_id)
    }

    /// Get a mutable reference to a task by ID
    pub fn get_task_mut(&mut self, task_id: TaskId) -> Option<&mut Task> {
        self.tasks.get_mut(&task_id)
    }

    /// Get all tasks
    pub fn get_all_tasks(&self) -> impl Iterator<Item = &Task> {
        self.tasks.values()
    }

    /// Get tasks by status
    pub fn get_tasks_by_status(&self, status_check: fn(&TaskStatus) -> bool) -> Vec<&Task> {
        self.tasks
            .values()
            .filter(|task| status_check(&task.status))
            .collect()
    }

    /// Get pending tasks
    pub fn get_pending_tasks(&self) -> Vec<&Task> {
        self.get_tasks_by_status(|status| matches!(status, TaskStatus::Pending))
    }

    /// Get completed tasks
    pub fn get_completed_tasks(&self) -> Vec<&Task> {
        self.get_tasks_by_status(|status| matches!(status, TaskStatus::Completed { .. }))
    }

    /// Get tasks in progress
    pub fn get_in_progress_tasks(&self) -> Vec<&Task> {
        self.get_tasks_by_status(|status| matches!(status, TaskStatus::InProgress { .. }))
    }

    /// Get tasks by priority
    pub fn get_tasks_by_priority(&self, priority: &PriorityLevel) -> Vec<&Task> {
        self.tasks
            .values()
            .filter(|task| &task.priority == priority)
            .collect()
    }

    /// Get tasks by assignee
    pub fn get_tasks_by_assignee(&self, assignee: &str) -> Vec<&Task> {
        self.tasks
            .values()
            .filter(|task| {
                task.assignee
                    .as_ref()
                    .map_or(false, |a| a == assignee)
            })
            .collect()
    }

    /// Get tasks by tag
    pub fn get_tasks_by_tag(&self, tag: &str) -> Vec<&Task> {
        self.tasks
            .values()
            .filter(|task| task.has_tag(tag))
            .collect()
    }

    /// Complete a task
    pub fn complete_task(&mut self, task_id: TaskId) -> Result<(), TaskManagerError> {
        let task = self.tasks
            .get_mut(&task_id)
            .ok_or(TaskManagerError::TaskNotFound(task_id))?;

        let old_status = task.status.clone();
        task.complete()?;
        self.update_statistics_for_status_change(&old_status, &task.status);

        Ok(())
    }

    /// Start a task
    pub fn start_task(&mut self, task_id: TaskId) -> Result<(), TaskManagerError> {
        let task = self.tasks
            .get_mut(&task_id)
            .ok_or(TaskManagerError::TaskNotFound(task_id))?;

        let old_status = task.status.clone();
        task.start()?;
        self.update_statistics_for_status_change(&old_status, &task.status);

        Ok(())
    }

    /// Cancel a task
    pub fn cancel_task(&mut self, task_id: TaskId, reason: String) -> Result<(), TaskManagerError> {
        let task = self.tasks
            .get_mut(&task_id)
            .ok_or(TaskManagerError::TaskNotFound(task_id))?;

        let old_status = task.status.clone();
        task.cancel(reason)?;
        self.update_statistics_for_status_change(&old_status, &task.status);

        Ok(())
    }

    /// Delete a task
    pub fn delete_task(&mut self, task_id: TaskId) -> Result<Task, TaskManagerError> {
        let task = self.tasks
            .remove(&task_id)
            .ok_or(TaskManagerError::TaskNotFound(task_id))?;

        self.update_statistics(&task, false);
        Ok(task)
    }

    /// Get comprehensive statistics
    pub fn get_statistics(&self) -> TaskStatistics {
        let total_tasks = self.tasks.len();
        let completed_tasks = self.get_completed_tasks().len();
        let pending_tasks = self.get_pending_tasks().len();
        let in_progress_tasks = self.get_in_progress_tasks().len();

        let completion_rate = if total_tasks > 0 {
            completed_tasks as f64 / total_tasks as f64 * 100.0
        } else {
            0.0
        };

        // Calculate average completion time
        let completed_durations: Vec<Duration> = self.get_completed_tasks()
            .iter()
            .filter_map(|task| task.duration())
            .collect();

        let average_completion_time = if !completed_durations.is_empty() {
            let total_duration: Duration = completed_durations.iter().sum();
            Some(total_duration / completed_durations.len() as u32)
        } else {
            None
        };

        // Priority distribution
        let mut priority_counts = HashMap::new();
        for task in self.tasks.values() {
            *priority_counts.entry(task.priority.clone()).or_insert(0) += 1;
        }

        // Tag frequency
        let mut tag_counts = HashMap::new();
        for task in self.tasks.values() {
            for tag in task.tags() {
                *tag_counts.entry(tag.clone()).or_insert(0) += 1;
            }
        }

        TaskStatistics {
            total_tasks,
            completed_tasks,
            pending_tasks,
            in_progress_tasks,
            completion_rate,
            average_completion_time,
            priority_distribution: priority_counts,
            tag_frequency: tag_counts,
        }
    }

    /// Update statistics when adding/removing tasks
    fn update_statistics(&mut self, task: &Task, adding: bool) {
        let delta = if adding { 1 } else { -1 };
        
        *self.task_count_by_priority
            .entry(task.priority.clone())
            .or_insert(0) = (self.task_count_by_priority
                .get(&task.priority)
                .unwrap_or(&0) + delta)
                .max(0) as usize;

        let status_key = self.status_to_string(&task.status);
        *self.task_count_by_status
            .entry(status_key)
            .or_insert(0) = (self.task_count_by_status
                .get(&self.status_to_string(&task.status))
                .unwrap_or(&0) + delta)
                .max(0) as usize;
    }

    /// Update statistics when task status changes
    fn update_statistics_for_status_change(&mut self, old_status: &TaskStatus, new_status: &TaskStatus) {
        let old_key = self.status_to_string(old_status);
        let new_key = self.status_to_string(new_status);

        if let Some(count) = self.task_count_by_status.get_mut(&old_key) {
            *count = count.saturating_sub(1);
        }

        *self.task_count_by_status.entry(new_key).or_insert(0) += 1;
    }

    /// Convert status to string for statistics
    fn status_to_string(&self, status: &TaskStatus) -> String {
        match status {
            TaskStatus::Pending => "pending".to_string(),
            TaskStatus::InProgress { .. } => "in_progress".to_string(),
            TaskStatus::Completed { .. } => "completed".to_string(),
            TaskStatus::Cancelled { .. } => "cancelled".to_string(),
        }
    }

    /// Find tasks matching a predicate
    pub fn find_tasks<F>(&self, predicate: F) -> Vec<&Task>
    where
        F: Fn(&Task) -> bool,
    {
        self.tasks.values().filter(|task| predicate(task)).collect()
    }

    /// Bulk operations
    pub fn complete_multiple_tasks(&mut self, task_ids: Vec<TaskId>) -> Vec<Result<(), TaskManagerError>> {
        task_ids
            .into_iter()
            .map(|id| self.complete_task(id))
            .collect()
    }
}

impl Default for TaskManager {
    fn default() -> Self {
        Self::new()
    }
}

// Statistics structure
#[derive(Debug, Clone)]
pub struct TaskStatistics {
    pub total_tasks: usize,
    pub completed_tasks: usize,
    pub pending_tasks: usize,
    pub in_progress_tasks: usize,
    pub completion_rate: f64,
    pub average_completion_time: Option<Duration>,
    pub priority_distribution: HashMap<PriorityLevel, usize>,
    pub tag_frequency: HashMap<String, usize>,
}

impl Display for TaskStatistics {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        writeln!(f, "Task Statistics:")?;
        writeln!(f, "  Total Tasks: {}", self.total_tasks)?;
        writeln!(f, "  Completed: {}", self.completed_tasks)?;
        writeln!(f, "  Pending: {}", self.pending_tasks)?;
        writeln!(f, "  In Progress: {}", self.in_progress_tasks)?;
        writeln!(f, "  Completion Rate: {:.1}%", self.completion_rate)?;

        if let Some(avg_time) = self.average_completion_time {
            writeln!(f, "  Average Completion Time: {:?}", avg_time)?;
        }

        writeln!(f, "  Priority Distribution:")?;
        for (priority, count) in &self.priority_distribution {
            writeln!(f, "    {}: {}", priority, count)?;
        }

        if !self.tag_frequency.is_empty() {
            writeln!(f, "  Top Tags:")?;
            let mut tags: Vec<_> = self.tag_frequency.iter().collect();
            tags.sort_by(|a, b| b.1.cmp(a.1));
            for (tag, count) in tags.iter().take(5) {
                writeln!(f, "    {}: {}", tag, count)?;
            }
        }

        Ok(())
    }
}

// Utility functions
/// Calculate the nth Fibonacci number using recursion with memoization
fn fibonacci(n: u64) -> u64 {
    fn fib_helper(n: u64, memo: &mut HashMap<u64, u64>) -> u64 {
        if let Some(&result) = memo.get(&n) {
            return result;
        }

        let result = match n {
            0 => 0,
            1 => 1,
            _ => fib_helper(n - 1, memo) + fib_helper(n - 2, memo),
        };

        memo.insert(n, result);
        result
    }

    let mut memo = HashMap::new();
    fib_helper(n, &mut memo)
}

/// Calculate Fibonacci number iteratively for better performance
fn fibonacci_iterative(n: u64) -> u64 {
    if n <= 1 {
        return n;
    }

    let mut a = 0;
    let mut b = 1;

    for _ in 2..=n {
        let temp = a + b;
        a = b;
        b = temp;
    }

    b
}

/// Generic function to process collections with transformations
fn process_collection<T, U, F>(collection: Vec<T>, transform: F) -> Vec<U>
where
    F: Fn(T) -> U,
{
    collection.into_iter().map(transform).collect()
}

/// Analyze a collection of numbers
#[derive(Debug, Clone)]
pub struct NumberAnalysis {
    pub count: usize,
    pub sum: f64,
    pub mean: f64,
    pub median: f64,
    pub min: f64,
    pub max: f64,
    pub variance: f64,
    pub std_dev: f64,
}

fn analyze_numbers(numbers: &[f64]) -> Option<NumberAnalysis> {
    if numbers.is_empty() {
        return None;
    }

    let count = numbers.len();
    let sum: f64 = numbers.iter().sum();
    let mean = sum / count as f64;

    let mut sorted_numbers = numbers.to_vec();
    sorted_numbers.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let median = if count % 2 == 0 {
        (sorted_numbers[count / 2 - 1] + sorted_numbers[count / 2]) / 2.0
    } else {
        sorted_numbers[count / 2]
    };

    let variance = numbers
        .iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>() / count as f64;

    let std_dev = variance.sqrt();

    Some(NumberAnalysis {
        count,
        sum,
        mean,
        median,
        min: sorted_numbers[0],
        max: sorted_numbers[count - 1],
        variance,
        std_dev,
    })
}

/// Complex function with advanced Rust features
fn complex_processing_pipeline<T, F, P>(
    data: Vec<T>,
    filter_predicate: F,
    processor: P,
) -> (Vec<T>, Duration)
where
    T: Clone,
    F: Fn(&T) -> bool,
    P: Fn(Vec<T>) -> Vec<T>,
{
    let start_time = Instant::now();

    let filtered_data: Vec<T> = data.into_iter().filter(filter_predicate).collect();
    let processed_data = processor(filtered_data);

    let processing_time = start_time.elapsed();

    (processed_data, processing_time)
}

/// Thread-safe task manager using Arc and Mutex
pub type ThreadSafeTaskManager = Arc<Mutex<TaskManager>>;

pub fn create_thread_safe_manager() -> ThreadSafeTaskManager {
    Arc::new(Mutex::new(TaskManager::new()))
}

/// Simulate concurrent task operations
pub fn simulate_concurrent_operations(manager: ThreadSafeTaskManager) -> Vec<thread::JoinHandle<()>> {
    let mut handles = Vec::new();

    // Create tasks concurrently
    for i in 0..5 {
        let manager_clone = Arc::clone(&manager);
        let handle = thread::spawn(move || {
            let mut mgr = manager_clone.lock().unwrap();
            let task_title = format!("Concurrent Task {}", i);
            if let Ok(task_id) = mgr.create_task(task_title) {
                println!("Created task {} with ID {}", i, task_id);
                
                // Simulate some work
                thread::sleep(Duration::from_millis(100));
                
                // Complete the task
                if mgr.complete_task(task_id).is_ok() {
                    println!("Completed task {}", task_id);
                }
            }
        });
        handles.push(handle);
    }

    handles
}

/// Main demonstration function
fn main() -> Result<(), Box<dyn Error>> {
    println!("Advanced Rust Task Manager Demo");
    println!("={}", "=".repeat(40));

    // Create task manager
    let mut manager = TaskManager::new();

    // Create various tasks
    let task1_id = manager.create_task_with_options(
        "Learn Rust".to_string(),
        Some("Master Rust programming concepts".to_string()),
        PriorityLevel::High,
        Some("developer".to_string()),
        vec!["learning".to_string(), "rust".to_string()],
    )?;

    let task2_id = manager.create_task_with_options(
        "Code Review".to_string(),
        Some("Review pull request #456".to_string()),
        PriorityLevel::Medium,
        Some("senior_dev".to_string()),
        vec!["review".to_string(), "code".to_string()],
    )?;

    let task3_id = manager.create_task_with_options(
        "Deploy to Production".to_string(),
        Some("Deploy version 3.2.1 to production".to_string()),
        PriorityLevel::Critical,
        Some("devops".to_string()),
        vec!["deploy".to_string(), "production".to_string()],
    )?;

    let task4_id = manager.create_task("Update Documentation".to_string())?;

    println!("Created {} tasks", manager.get_all_tasks().count());

    // Start and complete some tasks
    manager.start_task(task1_id)?;
    thread::sleep(Duration::from_millis(100));
    manager.complete_task(task1_id)?;

    manager.start_task(task2_id)?;
    thread::sleep(Duration::from_millis(50));
    manager.complete_task(task2_id)?;

    manager.cancel_task(task4_id, "Postponed to next sprint".to_string())?;

    // Display comprehensive statistics
    let stats = manager.get_statistics();
    println!("\n{}", stats);

    // Test utility functions
    println!("\nUtility Function Tests:");
    println!("Fibonacci(20) = {}", fibonacci(20));
    println!("Fibonacci Iterative(20) = {}", fibonacci_iterative(20));

    // Test number analysis
    let test_numbers = vec![1.0, 5.0, 3.0, 9.0, 2.0, 7.0, 4.0, 8.0, 6.0, 10.0];
    if let Some(analysis) = analyze_numbers(&test_numbers) {
        println!("\nNumber analysis for {:?}:", test_numbers);
        println!("  Count: {}", analysis.count);
        println!("  Sum: {:.2}", analysis.sum);
        println!("  Mean: {:.2}", analysis.mean);
        println!("  Median: {:.2}", analysis.median);
        println!("  Min: {:.2}", analysis.min);
        println!("  Max: {:.2}", analysis.max);
        println!("  Variance: {:.2}", analysis.variance);
        println!("  Std Dev: {:.2}", analysis.std_dev);
    }

    // Test complex processing pipeline
    let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let (processed, duration) = complex_processing_pipeline(
        data,
        |&x| x % 2 == 0, // Filter even numbers
        |mut vec| {
            vec.iter_mut().for_each(|x| *x *= 2); // Double each number
            vec
        },
    );

    println!("\nComplex Processing Results:");
    println!("  Processed data: {:?}", processed);
    println!("  Processing time: {:?}", duration);

    // Demonstrate concurrent operations
    println!("\nTesting concurrent operations...");
    let thread_safe_manager = create_thread_safe_manager();
    let handles = simulate_concurrent_operations(thread_safe_manager.clone());

    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }

    // Display final statistics from concurrent operations
    let final_stats = thread_safe_manager.lock().unwrap().get_statistics();
    println!("\nFinal Statistics after concurrent operations:");
    println!("{}", final_stats);

    println!("\nDemo completed successfully!");
    Ok(())
}