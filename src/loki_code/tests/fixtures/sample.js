/**
 * Sample JavaScript file for testing code analysis functionality.
 * 
 * This file contains various JavaScript constructs to test the completeness
 * of the Tree-sitter parsing and analysis capabilities.
 */

// Import statements (ES6 modules)
const fs = require('fs');
const path = require('path');

// Constants and enums
const PRIORITY_LEVELS = {
    LOW: 1,
    MEDIUM: 2,
    HIGH: 3,
    CRITICAL: 4
};

const STATUS = {
    PENDING: 'pending',
    IN_PROGRESS: 'in_progress',
    COMPLETED: 'completed',
    CANCELLED: 'cancelled'
};

/**
 * Task class representing a task with priority and status
 */
class Task {
    /**
     * Create a new task
     * @param {number} id - Task ID
     * @param {string} title - Task title
     * @param {string} description - Task description
     * @param {number} priority - Task priority level
     */
    constructor(id, title, description = null, priority = PRIORITY_LEVELS.MEDIUM) {
        this.id = id;
        this.title = title;
        this.description = description;
        this.priority = priority;
        this.status = STATUS.PENDING;
        this.createdAt = new Date();
        this.completedAt = null;
    }

    /**
     * Mark this task as completed
     */
    markComplete() {
        this.status = STATUS.COMPLETED;
        this.completedAt = new Date();
    }

    /**
     * Update the task priority
     * @param {number} newPriority - New priority level
     */
    updatePriority(newPriority) {
        if (Object.values(PRIORITY_LEVELS).includes(newPriority)) {
            this.priority = newPriority;
        } else {
            throw new Error('Invalid priority level');
        }
    }

    /**
     * Get task duration in milliseconds
     * @returns {number|null} Duration or null if not completed
     */
    getDuration() {
        if (this.completedAt) {
            return this.completedAt.getTime() - this.createdAt.getTime();
        }
        return null;
    }

    /**
     * Convert task to JSON representation
     * @returns {Object} Task as plain object
     */
    toJSON() {
        return {
            id: this.id,
            title: this.title,
            description: this.description,
            priority: this.priority,
            status: this.status,
            createdAt: this.createdAt.toISOString(),
            completedAt: this.completedAt ? this.completedAt.toISOString() : null,
            duration: this.getDuration()
        };
    }
}

/**
 * TaskManager class for managing a collection of tasks
 */
class TaskManager {
    /**
     * Initialize an empty task manager
     */
    constructor() {
        this.tasks = new Map();
        this.nextId = 1;
        this.listeners = new Map();
    }

    /**
     * Add a new task to the manager
     * @param {string} title - Task title
     * @param {string} description - Optional task description
     * @param {number} priority - Task priority level
     * @returns {number} The ID of the newly created task
     */
    addTask(title, description = null, priority = PRIORITY_LEVELS.MEDIUM) {
        const taskId = this.nextId++;
        const task = new Task(taskId, title, description, priority);
        
        this.tasks.set(taskId, task);
        this.emit('taskAdded', task);
        
        return taskId;
    }

    /**
     * Get a task by ID
     * @param {number} taskId - Task ID
     * @returns {Task|null} Task or null if not found
     */
    getTask(taskId) {
        return this.tasks.get(taskId) || null;
    }

    /**
     * Mark a task as completed
     * @param {number} taskId - Task ID
     * @returns {boolean} True if task was found and completed
     */
    completeTask(taskId) {
        const task = this.getTask(taskId);
        if (task) {
            task.markComplete();
            this.emit('taskCompleted', task);
            return true;
        }
        return false;
    }

    /**
     * Get all tasks with the specified priority
     * @param {number} priority - Priority level
     * @returns {Task[]} Array of tasks with specified priority
     */
    getTasksByPriority(priority) {
        return Array.from(this.tasks.values())
            .filter(task => task.priority === priority);
    }

    /**
     * Get all tasks with the specified status
     * @param {string} status - Task status
     * @returns {Task[]} Array of tasks with specified status
     */
    getTasksByStatus(status) {
        return Array.from(this.tasks.values())
            .filter(task => task.status === status);
    }

    /**
     * Get all pending tasks
     * @returns {Task[]} Array of pending tasks
     */
    getPendingTasks() {
        return this.getTasksByStatus(STATUS.PENDING);
    }

    /**
     * Get all completed tasks
     * @returns {Task[]} Array of completed tasks
     */
    getCompletedTasks() {
        return this.getTasksByStatus(STATUS.COMPLETED);
    }

    /**
     * Delete a task by ID
     * @param {number} taskId - Task ID
     * @returns {boolean} True if task was found and deleted
     */
    deleteTask(taskId) {
        const task = this.getTask(taskId);
        if (task) {
            this.tasks.delete(taskId);
            this.emit('taskDeleted', task);
            return true;
        }
        return false;
    }

    /**
     * Update the priority of a task
     * @param {number} taskId - Task ID
     * @param {number} priority - New priority level
     * @returns {boolean} True if task was found and updated
     */
    updateTaskPriority(taskId, priority) {
        const task = this.getTask(taskId);
        if (task) {
            try {
                task.updatePriority(priority);
                this.emit('taskUpdated', task);
                return true;
            } catch (error) {
                console.error('Failed to update task priority:', error.message);
            }
        }
        return false;
    }

    /**
     * Get statistics about the tasks
     * @returns {Object} Statistics object
     */
    getStatistics() {
        const totalTasks = this.tasks.size;
        const completedTasks = this.getCompletedTasks().length;
        const pendingTasks = this.getPendingTasks().length;
        
        const completionRate = totalTasks > 0 ? completedTasks / totalTasks : 0;
        
        const priorityCounts = {};
        Object.keys(PRIORITY_LEVELS).forEach(priority => {
            const level = PRIORITY_LEVELS[priority];
            priorityCounts[priority] = this.getTasksByPriority(level).length;
        });

        const statusCounts = {};
        Object.keys(STATUS).forEach(status => {
            statusCounts[status] = this.getTasksByStatus(STATUS[status]).length;
        });
        
        return {
            totalTasks,
            completedTasks,
            pendingTasks,
            completionRate: Math.round(completionRate * 100) / 100,
            priorityCounts,
            statusCounts
        };
    }

    /**
     * Add event listener
     * @param {string} event - Event name
     * @param {Function} callback - Callback function
     */
    on(event, callback) {
        if (!this.listeners.has(event)) {
            this.listeners.set(event, []);
        }
        this.listeners.get(event).push(callback);
    }

    /**
     * Emit event to all listeners
     * @param {string} event - Event name
     * @param {*} data - Event data
     */
    emit(event, data) {
        if (this.listeners.has(event)) {
            this.listeners.get(event).forEach(callback => {
                try {
                    callback(data);
                } catch (error) {
                    console.error(`Error in event listener for ${event}:`, error);
                }
            });
        }
    }

    /**
     * Export tasks to JSON
     * @returns {string} JSON string representation of all tasks
     */
    exportToJSON() {
        const tasksArray = Array.from(this.tasks.values()).map(task => task.toJSON());
        return JSON.stringify(tasksArray, null, 2);
    }

    /**
     * Import tasks from JSON
     * @param {string} jsonString - JSON string containing tasks
     * @returns {number} Number of tasks imported
     */
    importFromJSON(jsonString) {
        try {
            const tasksData = JSON.parse(jsonString);
            let importedCount = 0;

            tasksData.forEach(taskData => {
                const task = new Task(
                    taskData.id || this.nextId++,
                    taskData.title,
                    taskData.description,
                    taskData.priority
                );
                
                if (taskData.status) {
                    task.status = taskData.status;
                }
                
                if (taskData.createdAt) {
                    task.createdAt = new Date(taskData.createdAt);
                }
                
                if (taskData.completedAt) {
                    task.completedAt = new Date(taskData.completedAt);
                }

                this.tasks.set(task.id, task);
                importedCount++;
            });

            return importedCount;
        } catch (error) {
            console.error('Failed to import tasks from JSON:', error);
            return 0;
        }
    }
}

/**
 * Calculate the nth Fibonacci number using recursion
 * @param {number} n - The position in the Fibonacci sequence
 * @returns {number} The nth Fibonacci number
 */
function fibonacci(n) {
    if (n < 0) {
        throw new Error('Fibonacci is not defined for negative numbers');
    } else if (n <= 1) {
        return n;
    } else {
        return fibonacci(n - 1) + fibonacci(n - 2);
    }
}

/**
 * Calculate Fibonacci number iteratively for better performance
 * @param {number} n - The position in the Fibonacci sequence
 * @returns {number} The nth Fibonacci number
 */
function fibonacciIterative(n) {
    if (n < 0) {
        throw new Error('Fibonacci is not defined for negative numbers');
    } else if (n <= 1) {
        return n;
    }
    
    let a = 0, b = 1;
    for (let i = 2; i <= n; i++) {
        [a, b] = [b, a + b];
    }
    
    return b;
}

/**
 * Process a list of numeric data with optional transformation
 * @param {number[]} data - Array of numbers to process
 * @param {Function} transformFunc - Optional function to transform each element
 * @returns {number[]} Processed data array
 */
function processData(data, transformFunc = null) {
    if (!Array.isArray(data) || data.length === 0) {
        return [];
    }
    
    let processed = [...data];
    
    if (transformFunc && typeof transformFunc === 'function') {
        processed = processed.map(transformFunc);
    }
    
    // Filter out any non-numeric results
    processed = processed.filter(x => typeof x === 'number' && !isNaN(x));
    
    return processed;
}

/**
 * Analyze an array of numbers and return statistical information
 * @param {number[]} numbers - Array of numbers to analyze
 * @returns {Object} Object with statistical measures
 */
function analyzeNumbers(numbers) {
    if (!Array.isArray(numbers) || numbers.length === 0) {
        return { count: 0 };
    }
    
    const count = numbers.length;
    const sum = numbers.reduce((acc, num) => acc + num, 0);
    const mean = sum / count;
    
    // Calculate variance
    const variance = numbers.reduce((acc, num) => acc + Math.pow(num - mean, 2), 0) / count;
    const stdDev = Math.sqrt(variance);
    
    const sortedNumbers = [...numbers].sort((a, b) => a - b);
    
    // Calculate median
    const mid = Math.floor(count / 2);
    const median = count % 2 === 0 
        ? (sortedNumbers[mid - 1] + sortedNumbers[mid]) / 2
        : sortedNumbers[mid];
    
    return {
        count,
        sum: Math.round(sum * 100) / 100,
        mean: Math.round(mean * 100) / 100,
        median: Math.round(median * 100) / 100,
        min: Math.min(...numbers),
        max: Math.max(...numbers),
        variance: Math.round(variance * 100) / 100,
        stdDev: Math.round(stdDev * 100) / 100
    };
}

/**
 * A function with complex nested structure for testing parsing
 * @returns {Array} Array of results from nested operations
 */
function complexNestedFunction() {
    // Inner function with closure
    function innerFunction1(x) {
        function deeplyNested(y) {
            if (y > 0) {
                return x + y;
            } else {
                return x - y;
            }
        }
        return deeplyNested;
    }
    
    // Another inner function with complex logic
    function innerFunction2() {
        const localVar = "local";
        const results = [];
        
        for (let i = 0; i < 3; i++) {
            if (i % 2 === 0) {
                console.log(`Even: ${i}`);
                results.push(`even_${i}`);
            } else {
                console.log(`Odd: ${i}`);
                results.push(`odd_${i}`);
            }
        }
        
        return { localVar, results };
    }
    
    // Complex control flow with error handling
    const result = [];
    
    try {
        const func = innerFunction1(10);
        result.push(func(5));
        result.push(func(-3));
        
        const innerResult = innerFunction2();
        result.push(innerResult.results.length);
        
        // Async-like operation simulation
        const asyncResult = new Promise((resolve) => {
            setTimeout(() => resolve("async_result"), 100);
        });
        
        result.push("promise_created");
        
    } catch (error) {
        console.error('Error in complex function:', error);
        result.push(0);
    } finally {
        console.log('Complex function completed');
    }
    
    return result;
}

/**
 * Utility function for array operations
 * @param {Array} arr - Input array
 * @param {Function} predicate - Filter predicate
 * @returns {Array} Filtered and processed array
 */
const arrayUtils = {
    /**
     * Filter and map array in one operation
     */
    filterMap(arr, predicate, mapper) {
        return arr.filter(predicate).map(mapper);
    },
    
    /**
     * Group array elements by a key function
     */
    groupBy(arr, keyFunc) {
        return arr.reduce((groups, item) => {
            const key = keyFunc(item);
            if (!groups[key]) {
                groups[key] = [];
            }
            groups[key].push(item);
            return groups;
        }, {});
    },
    
    /**
     * Find unique elements in array
     */
    unique(arr) {
        return [...new Set(arr)];
    }
};

/**
 * Main function demonstrating the task manager functionality
 */
function main() {
    console.log('Task Manager Demo');
    console.log('='.repeat(40));
    
    // Create task manager
    const manager = new TaskManager();
    
    // Add event listeners
    manager.on('taskAdded', (task) => {
        console.log(`Task added: ${task.title}`);
    });
    
    manager.on('taskCompleted', (task) => {
        console.log(`Task completed: ${task.title} (Duration: ${task.getDuration()}ms)`);
    });
    
    // Add some tasks
    const task1Id = manager.addTask('Learn JavaScript', 'Study modern JavaScript features', PRIORITY_LEVELS.HIGH);
    const task2Id = manager.addTask('Buy groceries', 'Get milk, bread, and eggs', PRIORITY_LEVELS.MEDIUM);
    const task3Id = manager.addTask('Exercise', 'Go for a 30-minute run', PRIORITY_LEVELS.LOW);
    const task4Id = manager.addTask('Fix bug', 'Resolve critical production issue', PRIORITY_LEVELS.CRITICAL);
    
    console.log(`Added tasks with IDs: ${task1Id}, ${task2Id}, ${task3Id}, ${task4Id}`);
    
    // Complete some tasks
    setTimeout(() => {
        manager.completeTask(task2Id);
        manager.completeTask(task3Id);
        
        // Display statistics
        const stats = manager.getStatistics();
        console.log('\nTask Statistics:');
        Object.entries(stats).forEach(([key, value]) => {
            if (typeof value === 'object') {
                console.log(`  ${key}:`);
                Object.entries(value).forEach(([subKey, subValue]) => {
                    console.log(`    ${subKey}: ${subValue}`);
                });
            } else {
                console.log(`  ${key}: ${value}`);
            }
        });
        
        // Test fibonacci
        console.log(`\nFibonacci(10) = ${fibonacci(10)}`);
        console.log(`Fibonacci iterative(10) = ${fibonacciIterative(10)}`);
        
        // Test data analysis
        const testNumbers = [1, 5, 3, 9, 2, 7, 4, 8, 6];
        const analysis = analyzeNumbers(testNumbers);
        console.log(`\nNumber analysis for [${testNumbers.join(', ')}]:`);
        Object.entries(analysis).forEach(([key, value]) => {
            console.log(`  ${key}: ${value}`);
        });
        
        // Test complex function
        const complexResult = complexNestedFunction();
        console.log(`\nComplex function result: [${complexResult.join(', ')}]`);
        
        // Test array utilities
        const testArray = [1, 2, 3, 4, 5, 6];
        const evenDoubled = arrayUtils.filterMap(testArray, x => x % 2 === 0, x => x * 2);
        console.log(`\nEven numbers doubled: [${evenDoubled.join(', ')}]`);
        
    }, 100);
}

// Export for module use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        Task,
        TaskManager,
        fibonacci,
        fibonacciIterative,
        processData,
        analyzeNumbers,
        complexNestedFunction,
        arrayUtils,
        PRIORITY_LEVELS,
        STATUS
    };
}

// Run main if this is the main module
if (require.main === module) {
    main();
}