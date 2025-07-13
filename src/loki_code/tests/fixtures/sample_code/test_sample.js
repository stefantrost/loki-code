/**
 * Sample JavaScript code for testing Tree-sitter analysis.
 */

class Calculator {
    constructor() {
        this.history = [];
    }
    
    add(a, b) {
        const result = a + b;
        this.history.push(`${a} + ${b} = ${result}`);
        return result;
    }
    
    multiply(a, b) {
        const result = a * b;
        this.history.push(`${a} * ${b} = ${result}`);
        return result;
    }
    
    getHistory() {
        return this.history;
    }
}

function fibonacci(n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

function complexFunction(data, options = {}) {
    const defaultOptions = { verbose: false, timeout: 30 };
    const config = { ...defaultOptions, ...options };
    
    const result = {
        status: "processing",
        inputSize: Object.keys(data).length,
        timestamp: new Date().toISOString()
    };
    
    // Process data
    for (const [key, value] of Object.entries(data)) {
        if (typeof value === 'string') {
            result[key] = value.toUpperCase();
        } else if (typeof value === 'number') {
            result[key] = value * 2;
        } else {
            result[key] = String(value);
        }
    }
    
    result.status = "completed";
    return result;
}

// Arrow functions and modern JS features
const processArray = (arr) => arr.map(item => item * 2).filter(item => item > 10);

const asyncFunction = async (delay = 1000) => {
    return new Promise(resolve => {
        setTimeout(() => resolve("async result"), delay);
    });
};

// Export for module testing
if (typeof module !== 'undefined') {
    module.exports = {
        Calculator,
        fibonacci,
        complexFunction,
        processArray,
        asyncFunction
    };
}