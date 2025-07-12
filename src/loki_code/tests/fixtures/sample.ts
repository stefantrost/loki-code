/**
 * Sample TypeScript file for testing code analysis functionality.
 * 
 * This file contains various TypeScript constructs to test the completeness
 * of the Tree-sitter parsing and analysis capabilities.
 */

import { EventEmitter } from 'events';

// Type definitions and interfaces
interface TaskData {
    id: number;
    title: string;
    description?: string;
    priority: PriorityLevel;
    status: TaskStatus;
    createdAt: Date;
    completedAt?: Date;
    tags?: string[];
    assignee?: string;
}

interface TaskManagerOptions {
    maxTasks?: number;
    autoSave?: boolean;
    saveInterval?: number;
}

interface StatisticsData {
    totalTasks: number;
    completedTasks: number;
    pendingTasks: number;
    completionRate: number;
    priorityCounts: Record<string, number>;
    statusCounts: Record<string, number>;
    averageCompletionTime?: number;
}

interface AnalysisResult {
    count: number;
    sum?: number;
    mean?: number;
    median?: number;
    min?: number;
    max?: number;
    variance?: number;
    stdDev?: number;
    quartiles?: [number, number, number];
}

// Type aliases and unions
type PriorityLevel = 'low' | 'medium' | 'high' | 'critical';
type TaskStatus = 'pending' | 'in_progress' | 'completed' | 'cancelled';
type TransformFunction<T, U> = (item: T) => U;
type PredicateFunction<T> = (item: T) => boolean;

// Enums
enum Priority {
    LOW = 1,
    MEDIUM = 2,
    HIGH = 3,
    CRITICAL = 4
}

enum Status {
    PENDING = 'pending',
    IN_PROGRESS = 'in_progress',
    COMPLETED = 'completed',
    CANCELLED = 'cancelled'
}

// Generic interfaces
interface Repository<T> {
    create(item: Omit<T, 'id'>): T;
    findById(id: number): T | null;
    findAll(): T[];
    update(id: number, updates: Partial<T>): T | null;
    delete(id: number): boolean;
}

interface Serializable {
    toJSON(): Record<string, any>;
    fromJSON(data: Record<string, any>): void;
}

// Abstract base class
abstract class BaseEntity implements Serializable {
    protected _id: number;
    protected _createdAt: Date;
    protected _updatedAt: Date;

    constructor(id: number) {
        this._id = id;
        this._createdAt = new Date();
        this._updatedAt = new Date();
    }

    get id(): number {
        return this._id;
    }

    get createdAt(): Date {
        return this._createdAt;
    }

    get updatedAt(): Date {
        return this._updatedAt;
    }

    protected touch(): void {
        this._updatedAt = new Date();
    }

    abstract toJSON(): Record<string, any>;
    abstract fromJSON(data: Record<string, any>): void;
}

/**
 * Task class representing a task with TypeScript features
 */
class Task extends BaseEntity {
    private _title: string;
    private _description?: string;
    private _priority: PriorityLevel;
    private _status: TaskStatus;
    private _completedAt?: Date;
    private _tags: Set<string>;
    private _assignee?: string;

    constructor(
        id: number,
        title: string,
        options: {
            description?: string;
            priority?: PriorityLevel;
            tags?: string[];
            assignee?: string;
        } = {}
    ) {
        super(id);
        this._title = title;
        this._description = options.description;
        this._priority = options.priority || 'medium';
        this._status = 'pending';
        this._tags = new Set(options.tags || []);
        this._assignee = options.assignee;
    }

    // Getters and setters with validation
    get title(): string {
        return this._title;
    }

    set title(value: string) {
        if (!value.trim()) {
            throw new Error('Title cannot be empty');
        }
        this._title = value.trim();
        this.touch();
    }

    get description(): string | undefined {
        return this._description;
    }

    set description(value: string | undefined) {
        this._description = value?.trim() || undefined;
        this.touch();
    }

    get priority(): PriorityLevel {
        return this._priority;
    }

    set priority(value: PriorityLevel) {
        const validPriorities: PriorityLevel[] = ['low', 'medium', 'high', 'critical'];
        if (!validPriorities.includes(value)) {
            throw new Error(`Invalid priority: ${value}`);
        }
        this._priority = value;
        this.touch();
    }

    get status(): TaskStatus {
        return this._status;
    }

    get completedAt(): Date | undefined {
        return this._completedAt;
    }

    get tags(): string[] {
        return Array.from(this._tags);
    }

    get assignee(): string | undefined {
        return this._assignee;
    }

    set assignee(value: string | undefined) {
        this._assignee = value?.trim() || undefined;
        this.touch();
    }

    /**
     * Mark this task as completed
     */
    markComplete(): void {
        this._status = 'completed';
        this._completedAt = new Date();
        this.touch();
    }

    /**
     * Mark this task as in progress
     */
    markInProgress(): void {
        this._status = 'in_progress';
        this.touch();
    }

    /**
     * Cancel this task
     */
    cancel(): void {
        this._status = 'cancelled';
        this.touch();
    }

    /**
     * Add a tag to this task
     */
    addTag(tag: string): void {
        if (tag.trim()) {
            this._tags.add(tag.trim().toLowerCase());
            this.touch();
        }
    }

    /**
     * Remove a tag from this task
     */
    removeTag(tag: string): boolean {
        const removed = this._tags.delete(tag.trim().toLowerCase());
        if (removed) {
            this.touch();
        }
        return removed;
    }

    /**
     * Check if task has a specific tag
     */
    hasTag(tag: string): boolean {
        return this._tags.has(tag.trim().toLowerCase());
    }

    /**
     * Get task duration in milliseconds
     */
    getDuration(): number | null {
        if (this._completedAt) {
            return this._completedAt.getTime() - this.createdAt.getTime();
        }
        return null;
    }

    /**
     * Check if task is overdue (for demonstration)
     */
    isOverdue(dueDate?: Date): boolean {
        if (!dueDate || this._status === 'completed') {
            return false;
        }
        return new Date() > dueDate;
    }

    /**
     * Convert task to JSON representation
     */
    toJSON(): Record<string, any> {
        return {
            id: this.id,
            title: this._title,
            description: this._description,
            priority: this._priority,
            status: this._status,
            createdAt: this.createdAt.toISOString(),
            updatedAt: this.updatedAt.toISOString(),
            completedAt: this._completedAt?.toISOString(),
            tags: this.tags,
            assignee: this._assignee,
            duration: this.getDuration()
        };
    }

    /**
     * Create task from JSON data
     */
    fromJSON(data: Record<string, any>): void {
        this._title = data.title || '';
        this._description = data.description;
        this._priority = data.priority || 'medium';
        this._status = data.status || 'pending';
        this._assignee = data.assignee;
        
        if (data.tags && Array.isArray(data.tags)) {
            this._tags = new Set(data.tags);
        }
        
        if (data.completedAt) {
            this._completedAt = new Date(data.completedAt);
        }
    }

    /**
     * Clone this task with optional modifications
     */
    clone(modifications: Partial<TaskData> = {}): Task {
        const cloned = new Task(0, this._title, {
            description: this._description,
            priority: this._priority,
            tags: this.tags,
            assignee: this._assignee
        });
        
        // Apply modifications
        Object.assign(cloned, modifications);
        
        return cloned;
    }
}

/**
 * Advanced TaskManager with TypeScript features and generics
 */
class TaskManager extends EventEmitter implements Repository<Task> {
    private readonly tasks: Map<number, Task>;
    private nextId: number;
    private readonly options: Required<TaskManagerOptions>;
    private saveTimer?: NodeJS.Timeout;

    constructor(options: TaskManagerOptions = {}) {
        super();
        this.tasks = new Map();
        this.nextId = 1;
        this.options = {
            maxTasks: options.maxTasks || 1000,
            autoSave: options.autoSave || false,
            saveInterval: options.saveInterval || 60000 // 1 minute
        };

        if (this.options.autoSave) {
            this.startAutoSave();
        }
    }

    /**
     * Create a new task
     */
    create(taskData: Omit<TaskData, 'id' | 'createdAt' | 'status'>): Task {
        if (this.tasks.size >= this.options.maxTasks) {
            throw new Error(`Maximum number of tasks (${this.options.maxTasks}) reached`);
        }

        const task = new Task(this.nextId++, taskData.title, {
            description: taskData.description,
            priority: taskData.priority || 'medium',
            tags: taskData.tags,
            assignee: taskData.assignee
        });

        this.tasks.set(task.id, task);
        this.emit('taskCreated', task);
        
        return task;
    }

    /**
     * Find task by ID
     */
    findById(id: number): Task | null {
        return this.tasks.get(id) || null;
    }

    /**
     * Find all tasks
     */
    findAll(): Task[] {
        return Array.from(this.tasks.values());
    }

    /**
     * Update a task
     */
    update(id: number, updates: Partial<TaskData>): Task | null {
        const task = this.findById(id);
        if (!task) {
            return null;
        }

        // Apply updates using setters for validation
        if (updates.title !== undefined) task.title = updates.title;
        if (updates.description !== undefined) task.description = updates.description;
        if (updates.priority !== undefined) task.priority = updates.priority;
        if (updates.assignee !== undefined) task.assignee = updates.assignee;

        this.emit('taskUpdated', task);
        return task;
    }

    /**
     * Delete a task
     */
    delete(id: number): boolean {
        const task = this.findById(id);
        if (task) {
            this.tasks.delete(id);
            this.emit('taskDeleted', task);
            return true;
        }
        return false;
    }

    /**
     * Find tasks with generic filtering
     */
    findWhere<K extends keyof TaskData>(
        predicate: PredicateFunction<Task>
    ): Task[] {
        return this.findAll().filter(predicate);
    }

    /**
     * Get tasks by priority
     */
    getTasksByPriority(priority: PriorityLevel): Task[] {
        return this.findWhere(task => task.priority === priority);
    }

    /**
     * Get tasks by status
     */
    getTasksByStatus(status: TaskStatus): Task[] {
        return this.findWhere(task => task.status === status);
    }

    /**
     * Get tasks by assignee
     */
    getTasksByAssignee(assignee: string): Task[] {
        return this.findWhere(task => task.assignee === assignee);
    }

    /**
     * Get tasks by tag
     */
    getTasksByTag(tag: string): Task[] {
        return this.findWhere(task => task.hasTag(tag));
    }

    /**
     * Get pending tasks
     */
    getPendingTasks(): Task[] {
        return this.getTasksByStatus('pending');
    }

    /**
     * Get completed tasks
     */
    getCompletedTasks(): Task[] {
        return this.getTasksByStatus('completed');
    }

    /**
     * Complete a task
     */
    completeTask(id: number): boolean {
        const task = this.findById(id);
        if (task && task.status !== 'completed') {
            task.markComplete();
            this.emit('taskCompleted', task);
            return true;
        }
        return false;
    }

    /**
     * Bulk complete multiple tasks
     */
    completeTasks(ids: number[]): number {
        let completedCount = 0;
        ids.forEach(id => {
            if (this.completeTask(id)) {
                completedCount++;
            }
        });
        return completedCount;
    }

    /**
     * Get comprehensive statistics
     */
    getStatistics(): StatisticsData {
        const allTasks = this.findAll();
        const completedTasks = this.getCompletedTasks();
        const pendingTasks = this.getPendingTasks();
        
        const totalTasks = allTasks.length;
        const completionRate = totalTasks > 0 ? completedTasks.length / totalTasks : 0;

        // Priority distribution
        const priorityCounts: Record<string, number> = {
            low: 0, medium: 0, high: 0, critical: 0
        };
        
        // Status distribution
        const statusCounts: Record<string, number> = {
            pending: 0, in_progress: 0, completed: 0, cancelled: 0
        };

        allTasks.forEach(task => {
            priorityCounts[task.priority]++;
            statusCounts[task.status]++;
        });

        // Calculate average completion time
        const completedWithDuration = completedTasks
            .map(task => task.getDuration())
            .filter((duration): duration is number => duration !== null);
        
        const averageCompletionTime = completedWithDuration.length > 0
            ? completedWithDuration.reduce((sum, duration) => sum + duration, 0) / completedWithDuration.length
            : undefined;

        return {
            totalTasks,
            completedTasks: completedTasks.length,
            pendingTasks: pendingTasks.length,
            completionRate: Math.round(completionRate * 10000) / 100, // Round to 2 decimal places
            priorityCounts,
            statusCounts,
            averageCompletionTime
        };
    }

    /**
     * Export tasks to JSON
     */
    exportToJSON(): string {
        const tasksData = this.findAll().map(task => task.toJSON());
        return JSON.stringify(tasksData, null, 2);
    }

    /**
     * Import tasks from JSON
     */
    importFromJSON(jsonString: string): number {
        try {
            const tasksData: any[] = JSON.parse(jsonString);
            let importedCount = 0;

            tasksData.forEach(taskData => {
                try {
                    const task = this.create({
                        title: taskData.title,
                        description: taskData.description,
                        priority: taskData.priority || 'medium',
                        tags: taskData.tags,
                        assignee: taskData.assignee
                    });
                    
                    // Restore additional data
                    task.fromJSON(taskData);
                    importedCount++;
                } catch (error) {
                    console.warn(`Failed to import task: ${error}`);
                }
            });

            return importedCount;
        } catch (error) {
            console.error('Failed to parse JSON data:', error);
            return 0;
        }
    }

    /**
     * Start auto-save functionality
     */
    private startAutoSave(): void {
        this.saveTimer = setInterval(() => {
            this.emit('autoSave', this.exportToJSON());
        }, this.options.saveInterval);
    }

    /**
     * Stop auto-save functionality
     */
    stopAutoSave(): void {
        if (this.saveTimer) {
            clearInterval(this.saveTimer);
            this.saveTimer = undefined;
        }
    }

    /**
     * Clean up resources
     */
    destroy(): void {
        this.stopAutoSave();
        this.removeAllListeners();
        this.tasks.clear();
    }
}

/**
 * Generic utility functions with TypeScript features
 */
namespace Utils {
    /**
     * Calculate the nth Fibonacci number using recursion with memoization
     */
    const fibonacciCache = new Map<number, number>();
    
    export function fibonacci(n: number): number {
        if (n < 0) {
            throw new Error('Fibonacci is not defined for negative numbers');
        }
        
        if (fibonacciCache.has(n)) {
            return fibonacciCache.get(n)!;
        }
        
        let result: number;
        if (n <= 1) {
            result = n;
        } else {
            result = fibonacci(n - 1) + fibonacci(n - 2);
        }
        
        fibonacciCache.set(n, result);
        return result;
    }

    /**
     * Calculate Fibonacci number iteratively
     */
    export function fibonacciIterative(n: number): number {
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
     * Generic array processing function
     */
    export function processArray<T, U>(
        data: T[],
        transformFunc?: TransformFunction<T, U>,
        filterFunc?: PredicateFunction<T | U>
    ): (T | U)[] {
        if (!Array.isArray(data) || data.length === 0) {
            return [];
        }
        
        let processed: (T | U)[] = [...data];
        
        if (transformFunc) {
            processed = (processed as T[]).map(transformFunc);
        }
        
        if (filterFunc) {
            processed = processed.filter(filterFunc);
        }
        
        return processed;
    }

    /**
     * Analyze an array of numbers with comprehensive statistics
     */
    export function analyzeNumbers(numbers: number[]): AnalysisResult {
        if (!Array.isArray(numbers) || numbers.length === 0) {
            return { count: 0 };
        }
        
        const count = numbers.length;
        const sum = numbers.reduce((acc, num) => acc + num, 0);
        const mean = sum / count;
        
        // Calculate variance and standard deviation
        const variance = numbers.reduce((acc, num) => {
            return acc + Math.pow(num - mean, 2);
        }, 0) / count;
        const stdDev = Math.sqrt(variance);
        
        // Sort for median and quartiles
        const sortedNumbers = [...numbers].sort((a, b) => a - b);
        
        // Calculate median
        const mid = Math.floor(count / 2);
        const median = count % 2 === 0 
            ? (sortedNumbers[mid - 1] + sortedNumbers[mid]) / 2
            : sortedNumbers[mid];
        
        // Calculate quartiles
        const q1Index = Math.floor(count / 4);
        const q3Index = Math.floor(3 * count / 4);
        const quartiles: [number, number, number] = [
            sortedNumbers[q1Index],
            median,
            sortedNumbers[q3Index]
        ];
        
        return {
            count,
            sum: Number(sum.toFixed(2)),
            mean: Number(mean.toFixed(2)),
            median: Number(median.toFixed(2)),
            min: Math.min(...numbers),
            max: Math.max(...numbers),
            variance: Number(variance.toFixed(2)),
            stdDev: Number(stdDev.toFixed(2)),
            quartiles
        };
    }

    /**
     * Generic grouping function
     */
    export function groupBy<T, K extends string | number>(
        array: T[],
        keySelector: (item: T) => K
    ): Record<K, T[]> {
        return array.reduce((groups, item) => {
            const key = keySelector(item);
            if (!groups[key]) {
                groups[key] = [];
            }
            groups[key].push(item);
            return groups;
        }, {} as Record<K, T[]>);
    }

    /**
     * Type-safe array utilities
     */
    export const ArrayUtils = {
        /**
         * Get unique elements preserving order
         */
        unique<T>(array: T[]): T[] {
            return Array.from(new Set(array));
        },

        /**
         * Chunk array into smaller arrays
         */
        chunk<T>(array: T[], size: number): T[][] {
            const chunks: T[][] = [];
            for (let i = 0; i < array.length; i += size) {
                chunks.push(array.slice(i, i + size));
            }
            return chunks;
        },

        /**
         * Flatten nested arrays
         */
        flatten<T>(array: (T | T[])[]): T[] {
            return array.reduce<T[]>((flat, item) => {
                return flat.concat(Array.isArray(item) ? this.flatten(item) : item);
            }, []);
        }
    };
}

/**
 * Complex function with advanced TypeScript features
 */
function complexTypeScriptFunction<T extends Record<string, any>>(
    data: T[],
    options: {
        sortBy?: keyof T;
        filterBy?: PredicateFunction<T>;
        groupBy?: keyof T;
        transform?: TransformFunction<T, any>;
    } = {}
): {
    processed: T[];
    groups?: Record<string, T[]>;
    statistics: {
        totalItems: number;
        filteredItems: number;
        processingTime: number;
    };
} {
    const startTime = performance.now();
    
    let processed = [...data];
    
    // Apply filtering
    if (options.filterBy) {
        processed = processed.filter(options.filterBy);
    }
    
    // Apply sorting
    if (options.sortBy) {
        processed.sort((a, b) => {
            const aVal = a[options.sortBy!];
            const bVal = b[options.sortBy!];
            
            if (aVal < bVal) return -1;
            if (aVal > bVal) return 1;
            return 0;
        });
    }
    
    // Apply transformation
    if (options.transform) {
        processed = processed.map(options.transform);
    }
    
    // Apply grouping
    let groups: Record<string, T[]> | undefined;
    if (options.groupBy) {
        groups = Utils.groupBy(processed, item => String(item[options.groupBy!]));
    }
    
    const endTime = performance.now();
    
    return {
        processed,
        groups,
        statistics: {
            totalItems: data.length,
            filteredItems: processed.length,
            processingTime: endTime - startTime
        }
    };
}

/**
 * Main demonstration function
 */
async function main(): Promise<void> {
    console.log('Advanced TypeScript Task Manager Demo');
    console.log('='.repeat(50));
    
    // Create task manager with options
    const manager = new TaskManager({
        maxTasks: 100,
        autoSave: true,
        saveInterval: 30000
    });
    
    // Add event listeners
    manager.on('taskCreated', (task: Task) => {
        console.log(`Task created: ${task.title} (ID: ${task.id})`);
    });
    
    manager.on('taskCompleted', (task: Task) => {
        const duration = task.getDuration();
        console.log(`Task completed: ${task.title} (Duration: ${duration}ms)`);
    });
    
    manager.on('autoSave', (jsonData: string) => {
        console.log('Auto-save triggered');
    });
    
    try {
        // Create various tasks
        const tasks = [
            manager.create({
                title: 'Learn TypeScript',
                description: 'Master advanced TypeScript features',
                priority: 'high',
                tags: ['learning', 'typescript'],
                assignee: 'developer'
            }),
            manager.create({
                title: 'Code Review',
                description: 'Review pull request #123',
                priority: 'medium',
                tags: ['review', 'code'],
                assignee: 'senior_dev'
            }),
            manager.create({
                title: 'Deploy to Production',
                description: 'Deploy version 2.1.0 to production',
                priority: 'critical',
                tags: ['deploy', 'production'],
                assignee: 'devops'
            }),
            manager.create({
                title: 'Update Documentation',
                description: 'Update API documentation',
                priority: 'low',
                tags: ['docs', 'api'],
                assignee: 'technical_writer'
            })
        ];
        
        console.log(`\nCreated ${tasks.length} tasks`);
        
        // Simulate task progression
        await new Promise(resolve => setTimeout(resolve, 100));
        
        // Complete some tasks
        manager.completeTask(tasks[1].id);
        await new Promise(resolve => setTimeout(resolve, 50));
        
        manager.completeTask(tasks[3].id);
        
        // Display comprehensive statistics
        const stats = manager.getStatistics();
        console.log('\nTask Statistics:');
        console.log(`  Total Tasks: ${stats.totalTasks}`);
        console.log(`  Completed: ${stats.completedTasks}`);
        console.log(`  Pending: ${stats.pendingTasks}`);
        console.log(`  Completion Rate: ${stats.completionRate}%`);
        
        if (stats.averageCompletionTime) {
            console.log(`  Average Completion Time: ${stats.averageCompletionTime.toFixed(2)}ms`);
        }
        
        console.log('\nPriority Distribution:');
        Object.entries(stats.priorityCounts).forEach(([priority, count]) => {
            console.log(`  ${priority}: ${count}`);
        });
        
        console.log('\nStatus Distribution:');
        Object.entries(stats.statusCounts).forEach(([status, count]) => {
            console.log(`  ${status}: ${count}`);
        });
        
        // Test utility functions
        console.log('\nUtility Function Tests:');
        console.log(`Fibonacci(15) = ${Utils.fibonacci(15)}`);
        console.log(`Fibonacci Iterative(15) = ${Utils.fibonacciIterative(15)}`);
        
        // Test number analysis
        const testNumbers = [1, 5, 3, 9, 2, 7, 4, 8, 6, 10];
        const analysis = Utils.analyzeNumbers(testNumbers);
        console.log(`\nNumber analysis for [${testNumbers.join(', ')}]:`);
        Object.entries(analysis).forEach(([key, value]) => {
            if (Array.isArray(value)) {
                console.log(`  ${key}: [${value.join(', ')}]`);
            } else {
                console.log(`  ${key}: ${value}`);
            }
        });
        
        // Test complex function
        const complexResult = complexTypeScriptFunction(tasks, {
            filterBy: (task: Task) => task.priority === 'high' || task.priority === 'critical',
            sortBy: 'priority',
            groupBy: 'assignee'
        });
        
        console.log('\nComplex Function Results:');
        console.log(`  Processed ${complexResult.statistics.filteredItems} out of ${complexResult.statistics.totalItems} tasks`);
        console.log(`  Processing time: ${complexResult.statistics.processingTime.toFixed(2)}ms`);
        
        if (complexResult.groups) {
            console.log('  Groups:');
            Object.entries(complexResult.groups).forEach(([assignee, groupTasks]) => {
                console.log(`    ${assignee}: ${groupTasks.length} task(s)`);
            });
        }
        
    } catch (error) {
        console.error('Error in main function:', error);
    } finally {
        // Clean up
        manager.destroy();
        console.log('\nCleanup completed');
    }
}

// Export for module use
export {
    Task,
    TaskManager,
    Utils,
    complexTypeScriptFunction,
    Priority,
    Status,
    type TaskData,
    type StatisticsData,
    type AnalysisResult,
    type PriorityLevel,
    type TaskStatus
};

// Run main if this is the main module
if (require.main === module) {
    main().catch(console.error);
}