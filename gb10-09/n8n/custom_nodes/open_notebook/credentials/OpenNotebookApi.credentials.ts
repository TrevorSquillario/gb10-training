import {
	IAuthenticateGeneric,
	ICredentialType,
	INodeProperties,
} from 'n8n-workflow';

export class OpenNotebookApi implements ICredentialType {
	name = 'openNotebookApi';
	displayName = 'Open Notebook API';
	documentationUrl = 'https://docs.opennotebook.io';
	properties: INodeProperties[] = [
		{
			displayName: 'API URL',
			name: 'apiUrl',
			type: 'string',
			default: 'http://localhost:5055',
			placeholder: 'http://localhost:5055',
			description: 'The base URL of your Open Notebook API',
		},
		{
			displayName: 'API Key',
			name: 'apiKey',
			type: 'string',
			typeOptions: {
				password: true,
			},
			default: '',
			description: 'Your Open Notebook API key',
		},
	];
	authenticate: IAuthenticateGeneric = {
		type: 'generic',
		properties: {
			headers: {
				Authorization: 'Bearer {{$credentials.apiKey}}',
			},
		},
	};
}
