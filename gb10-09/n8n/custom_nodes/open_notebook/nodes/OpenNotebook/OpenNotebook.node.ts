import {
	INodeType,
	INodeTypeDescription,
	ILoadOptionsFunctions,
	INodePropertyOptions,
} from 'n8n-workflow';

export class OpenNotebook implements INodeType {
	description: INodeTypeDescription = {
		displayName: 'Open Notebook',
		name: 'openNotebook',
		icon: 'file:opennotebook.svg',
		group: ['transform'],
		version: 1,
		subtitle: '={{$parameter["operation"] + ": " + $parameter["resource"]}}',
		description: 'Interact with Open Notebook API',
		defaults: {
			name: 'Open Notebook',
		},
		inputs: ['main'],
		outputs: ['main'],
		credentials: [
			{
				name: 'openNotebookApi',
				required: true,
			},
		],
		requestDefaults: {
			baseURL: '={{$credentials.apiUrl}}',
			headers: {
				Accept: 'application/json',
				'Content-Type': 'application/json',
			},
		},
		properties: [
			// Resource
			{
				displayName: 'Resource',
				name: 'resource',
				type: 'options',
				noDataExpression: true,
				options: [
					{
						name: 'Note',
						value: 'note',
					},
				],
				default: 'note',
			},
			// Operations
			{
				displayName: 'Operation',
				name: 'operation',
				type: 'options',
				noDataExpression: true,
				displayOptions: {
					show: {
						resource: ['note'],
					},
				},
				options: [
					{
						name: 'Create',
						value: 'create',
						action: 'Create a note',
						description: 'Create a new note in Open Notebook',
						routing: {
							request: {
								method: 'POST',
								url: '/api/notes',
							},
						},
					},
				],
				default: 'create',
			},
			// Fields for Create Note operation
			{
				displayName: 'Notebook',
				name: 'notebook_id',
				type: 'options',
				typeOptions: {
					loadOptionsMethod: 'getNotebooks',
				},
				required: true,
				displayOptions: {
					show: {
						resource: ['note'],
						operation: ['create'],
					},
				},
				default: '',
				description: 'The notebook to create the note in',
				routing: {
					request: {
						body: {
							notebook_id: '={{ $value }}',
						},
					},
				},
			},
			{
				displayName: 'Title',
				name: 'title',
				type: 'string',
				required: true,
				displayOptions: {
					show: {
						resource: ['note'],
						operation: ['create'],
					},
				},
				default: '',
				description: 'The title of the note',
				routing: {
					request: {
						body: {
							title: '={{ $value }}',
						},
					},
				},
			},
			{
				displayName: 'Content',
				name: 'content',
				type: 'string',
				typeOptions: {
					rows: 5,
				},
				required: true,
				displayOptions: {
					show: {
						resource: ['note'],
						operation: ['create'],
					},
				},
				default: '',
				description: 'The content of the note',
				routing: {
					request: {
						body: {
							content: '={{ $value }}',
						},
					},
				},
			},
			{
				displayName: 'Note Type',
				name: 'note_type',
				type: 'options',
				options: [
					{
						name: 'Human',
						value: 'human',
					},
					{
						name: 'AI',
						value: 'ai',
					},
					{
						name: 'System',
						value: 'system',
					},
				],
				displayOptions: {
					show: {
						resource: ['note'],
						operation: ['create'],
					},
				},
				default: 'human',
				description: 'The type of note',
				routing: {
					request: {
						body: {
							note_type: '={{ $value }}',
						},
					},
				},
			},
		],
	};

	methods = {
		loadOptions: {
			async getNotebooks(this: ILoadOptionsFunctions): Promise<INodePropertyOptions[]> {
				const credentials = await this.getCredentials('openNotebookApi');
				const apiUrl = credentials.apiUrl as string;
				const apiKey = credentials.apiKey as string;
				
				const response = await this.helpers.request({
					method: 'GET',
					url: `${apiUrl}/api/notebooks`,
					headers: {
						Authorization: `Bearer ${apiKey}`,
					},
					json: true,
				});

				const notebooks = response as Array<{
					id: string;
					name: string;
					description?: string;
				}>;

				return notebooks.map((notebook) => ({
					name: notebook.name,
					value: notebook.id,
					description: notebook.description,
				}));
			},
		},
	};
}
