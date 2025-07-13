/**
 * Sample TypeScript code for testing Tree-sitter analysis.
 */

interface CalculatorOptions {
    precision?: number;
    verbose?: boolean;
}

interface CalculationResult {
    result: number;
    operation: string;
    timestamp: Date;
}

class Calculator {
    private history: CalculationResult[] = [];
    private options: CalculatorOptions;
    
    constructor(options: CalculatorOptions = {}) {
        this.options = { precision: 2, verbose: false, ...options };
    }
    
    public add(a: number, b: number): number {
        const result = a + b;
        const calculation: CalculationResult = {
            result,
            operation: `${a} + ${b}`,
            timestamp: new Date()
        };
        this.history.push(calculation);
        return result;
    }
    
    public multiply(a: number, b: number): number {
        const result = a * b;
        const calculation: CalculationResult = {
            result,
            operation: `${a} * ${b}`,
            timestamp: new Date()
        };
        this.history.push(calculation);
        return result;
    }
    
    public getHistory(): CalculationResult[] {
        return [...this.history];
    }
}

function fibonacci(n: number): number {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

type DataItem = string | number | boolean | null;
type DataObject = Record<string, DataItem>;

interface ProcessingOptions {
    verbose?: boolean;
    timeout?: number;
}

interface ProcessingResult {
    status: "processing" | "completed" | "error";
    inputSize: number;
    timestamp: string;
    [key: string]: any;
}

function complexFunction(data: DataObject, options: ProcessingOptions = {}): ProcessingResult {
    const defaultOptions: ProcessingOptions = { verbose: false, timeout: 30 };
    const config = { ...defaultOptions, ...options };
    
    const result: ProcessingResult = {
        status: "processing",
        inputSize: Object.keys(data).length,
        timestamp: new Date().toISOString()
    };
    
    // Process data with type checking
    for (const [key, value] of Object.entries(data)) {
        if (typeof value === 'string') {
            result[key] = value.toUpperCase();
        } else if (typeof value === 'number') {
            result[key] = value * 2;
        } else if (typeof value === 'boolean') {
            result[key] = !value;
        } else {
            result[key] = String(value);
        }
    }
    
    result.status = "completed";
    return result;
}

// Generic function with constraints
function processItems<T extends { id: number }>(items: T[]): T[] {
    return items.filter(item => item.id > 0);
}

// Async function with proper typing
async function fetchData(url: string): Promise<{ data: any; status: number }> {
    // Simulated async operation
    return new Promise(resolve => {
        setTimeout(() => {
            resolve({ data: { message: "success" }, status: 200 });
        }, 100);
    });
}

export {
    Calculator,
    fibonacci,
    complexFunction,
    processItems,
    fetchData,
    type CalculatorOptions,
    type ProcessingResult
};