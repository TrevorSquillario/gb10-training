Write me a custom node for open_notebook api integration.

Refer to the documentation at https://docs.n8n.io/integrations/creating-nodes/build/declarative-style-node/

I need the following features to be implemented:

Create Note Node
This will have a dropdown populated from /api/notebooks. Expect the following response
[
  {
    "id": "string",
    "name": "string",
    "description": "string",
    "archived": true,
    "created": "string",
    "updated": "string",
    "source_count": 0,
    "note_count": 0
  }
]

It will POST to /api/notes using the following request
{
  "title": "string",
  "content": "string",
  "note_type": "human",
  "notebook_id": "string"
}

