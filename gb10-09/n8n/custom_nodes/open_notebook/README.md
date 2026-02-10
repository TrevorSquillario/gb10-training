# n8n-nodes-open-notebook

This is an n8n community node for integrating with the Open Notebook API.

[n8n](https://n8n.io/) is a [fair-code licensed](https://docs.n8n.io/reference/license/) workflow automation platform.

## Features

- **Create Note**: Create new notes in your Open Notebook notebooks

## Installation

Follow the [installation guide](https://docs.n8n.io/integrations/community-nodes/installation/) in the n8n community nodes documentation.

### Manual Installation

1. Navigate to your n8n custom nodes directory:
   ```bash
   cd ~/.n8n/custom
   ```

2. Clone or copy this repository into the custom directory

3. Install dependencies:
   ```bash
   cd n8n-nodes-open-notebook
   npm install
   ```

4. Build the node:
   ```bash
   npm run build
   ```

5. Restart n8n

## Credentials

To use this node, you'll need to set up Open Notebook API credentials:

1. **API URL**: The base URL of your Open Notebook API instance (e.g., `http://localhost:8000`)
2. **API Key**: Your Open Notebook API authentication key

## Operations

### Note

#### Create
Create a new note in a specified notebook.

**Parameters:**
- **Notebook**: Select the notebook where the note will be created (loaded dynamically from your Open Notebook instance)
- **Title**: The title of the note
- **Content**: The content/body of the note
- **Note Type**: Type of note (Human, AI, or System) - defaults to Human

## API Endpoints Used

- `GET /api/notebooks` - Fetches available notebooks for the dropdown
- `POST /api/notes` - Creates a new note

## Example Usage

1. Add the "Open Notebook" node to your workflow
2. Connect your Open Notebook API credentials
3. Select "Note" as the resource
4. Select "Create" as the operation
5. Choose a notebook from the dropdown
6. Enter the title and content for your note
7. Optionally change the note type (defaults to "human")

## Compatibility

Tested with n8n version 1.0.0 and later.

## Resources

- [n8n community nodes documentation](https://docs.n8n.io/integrations/community-nodes/)
- [Open Notebook API documentation](https://docs.opennotebook.io)

## License

[MIT](LICENSE)
