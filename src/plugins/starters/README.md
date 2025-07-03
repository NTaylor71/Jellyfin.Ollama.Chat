# Starter Plugins

This directory contains example plugins to help you get started with the plugin system.

## Available Starters

- `query_expander.py` - Example query embellisher plugin
- `embed_enhancer.py` - Example embed data embellisher plugin
- `faiss_logger.py` - Example FAISS CRUD plugin

## How to Use

1. Copy a starter plugin to your desired location
2. Modify the plugin class name and metadata
3. Implement your custom logic
4. The plugin will be automatically discovered and loaded

## Plugin Development Tips

- Use the `@plugin_decorator` for easy metadata setup
- Check hardware requirements in `resource_requirements`
- Implement proper error handling in your `execute` method
- Use the plugin's logger for debugging: `self._logger.info("message")`