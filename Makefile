.PHONY: all build test clean lint docs install help format check format-check

# Default target
all: build test

# Build all components
build:
	@echo "🔨 Building Rust components..."
	cargo build --release
	@echo "✅ Rust build complete"

# Run all tests
test:
	@echo "🧪 Running Rust tests..."
	cargo test -- --nocapture
	@echo "✅ Rust tests complete"

# Run tests with coverage
test-coverage:
	@echo "📊 Running tests with coverage..."
	cargo test --no-fail-fast
	@echo "✅ Coverage tests complete"

# Clean build artifacts
clean:
	@echo "🧹 Cleaning build artifacts..."
	cargo clean
	@echo "✅ Clean complete"

# Lint code
lint:
	@echo "🔍 Linting Rust code..."
	cargo clippy -- -D warnings
	@echo "✅ Linting complete"

# Format code
format:
	@echo "🎨 Formatting code..."
	cargo fmt
	@echo "✅ Formatting complete"

# Check formatting without making changes
format-check:
	@echo "🔍 Checking code format..."
	cargo fmt -- --check
	@echo "✅ Format check complete"

# Generate documentation
docs:
	@echo "📚 Generating documentation..."
	cargo doc --no-deps --document-private-items
	@echo "✅ Documentation generated"
	@echo "📖 Open with: cargo doc --no-deps --open"

# Run examples
run-examples:
	@echo "🚀 Running examples..."
	cargo run --release --bin equilibrium-daemon
	@echo "✅ Examples complete"

# Install project
install:
	@echo "📦 Installing project..."
	cargo install --path .
	@echo "✅ Installation complete"

# Check code
check:
	@echo "🔍 Checking code..."
	cargo check
	@echo "✅ Check complete"

# Run benchmarks
bench:
	@echo "⚡ Running benchmarks..."
	cargo bench
	@echo "✅ Benchmarks complete"

# Update dependencies
update:
	@echo "📦 Updating dependencies..."
	cargo update
	@echo "✅ Dependencies updated"

# Audit dependencies for security vulnerabilities
audit:
	@echo "🔒 Auditing dependencies..."
	cargo audit
	@echo "✅ Audit complete"

# Release build (optimized)
release: clean lint test
	@echo "🚀 Building release..."
	cargo build --release
	@echo "✅ Release build complete"

# Development build (with debug info)
dev:
	@echo "🔧 Building development version..."
	cargo build
	@echo "✅ Development build complete"

# Watch for changes and rebuild
watch:
	@echo "👀 Watching for changes..."
	cargo watch -x build -x test -x run

# Show help
help:
	@echo "📖 Equilibrium Tokens - Available Commands"
	@echo ""
	@echo "Building:"
	@echo "  make build         - Build all components"
	@echo "  make release       - Build optimized release version"
	@echo "  make dev           - Build development version"
	@echo ""
	@echo "Testing:"
	@echo "  make test          - Run all tests"
	@echo "  make test-coverage - Run tests with coverage"
	@echo "  make bench         - Run benchmarks"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint          - Lint code"
	@echo "  make format        - Format code"
	@echo "  make format-check  - Check code format"
	@echo "  make audit         - Audit dependencies"
	@echo ""
	@echo "Documentation:"
	@echo "  make docs          - Generate documentation"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean         - Clean build artifacts"
	@echo "  make update        - Update dependencies"
	@echo "  make install       - Install project locally"
	@echo ""
	@echo "Other:"
	@echo "  make watch         - Watch for changes and rebuild"
	@echo "  make help          - Show this help message"
