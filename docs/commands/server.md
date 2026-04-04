# server

Start the datasety REST API server for headless, remote dataset management and job execution.

The API provides endpoints to register datasets, browse their files, and trigger _any_ `datasety` command remotely using JSON payloads.

## Usage

```bash
datasety server --port 8080
```

_Note: If the requested port is occupied, the server will automatically find the next available port._

## REST API Endpoints (Base path: `/v1/`)

<details>
<summary><b>Endpoints</b></summary>

| Endpoint                            | Method | Description                                          |
| ----------------------------------- | ------ | ---------------------------------------------------- |
| `/v1/datasets`                      | POST   | Register a dataset                                   |
| `/v1/datasets`                      | GET    | List all datasets                                    |
| `/v1/datasets/<id>`                 | GET    | Get dataset info                                     |
| `/v1/datasets/<id>`                 | PATCH  | Update dataset name                                  |
| `/v1/datasets/<id>`                 | DELETE | Unregister dataset                                   |
| `/v1/datasets/<id>/files`           | GET    | List files (supports `?folder=&group=` query params) |
| `/v1/datasets/<id>/files/<path>`    | GET    | Download/serve a file                                |
| `/v1/datasets/<id>/caption/<path>`  | GET    | Get caption for image                                |
| `/v1/datasets/<id>/caption/<path>`  | PUT    | Save caption for image                               |
| `/v1/datasets/<id>/metadata/<path>` | PUT    | Update metadata.csv text                             |
| `/v1/jobs`                          | GET    | List all jobs                                        |
| `/v1/jobs`                          | POST   | Start a new job (run any datasety command)           |
| `/v1/jobs/<id>`                     | GET    | Get job status & output                              |
| `/v1/jobs/<id>`                     | DELETE | Cancel a running job                                 |
| `/v1/commands`                      | GET    | Get command schemas                                  |

</details>

### Datasets

#### Add a Dataset

Registers a directory for management. Automatically detects if it's an `audio`, `image`, `video`, etc. dataset.

`POST /v1/datasets`

**Request:**

```json
{
  "name": "My Portraits",
  "path": "/absolute/path/to/dataset"
}
```

**Response: 201 Created**

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "name": "My Portraits",
  "path": "/absolute/path/to/dataset",
  "type": "image",
  "created_at": 1709632800.123456
}
```

**Error Responses:**

`400 Bad Request` - Missing path:

```json
{
  "error": "Missing 'path' in payload"
}
```

`400 Bad Request` - Path doesn't exist:

```json
{
  "error": "Provided path does not exist or is not a directory"
}
```

`400 Bad Request` - Invalid JSON:

```json
{
  "error": "Invalid JSON payload"
}
```

---

#### List Datasets

`GET /v1/datasets`

**Response: 200 OK**

```json
{
  "datasets": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "name": "My Portraits",
      "path": "/absolute/path/to/dataset",
      "type": "image",
      "created_at": 1709632800.123456
    },
    {
      "id": "6ba7b810-9dad-11d1-80b4-00c04fd430c8",
      "name": "Audio Samples",
      "path": "/path/to/audio",
      "type": "audio",
      "created_at": 1709632900.789012
    }
  ]
}
```

---

#### Get Dataset Info

`GET /v1/datasets/<id>`

**Response: 200 OK**

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "name": "My Portraits",
  "path": "/absolute/path/to/dataset",
  "type": "image",
  "created_at": 1709632800.123456
}
```

**Error Response:**

`404 Not Found`:

```json
{
  "error": "Dataset not found"
}
```

---

#### Modify Dataset

`PATCH /v1/datasets/<id>`

**Request:**

```json
{ "name": "Updated Name" }
```

**Response: 200 OK**

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "name": "Updated Name",
  "path": "/absolute/path/to/dataset",
  "type": "image",
  "created_at": 1709632800.123456
}
```

**Error Responses:**

`400 Bad Request` - Invalid JSON:

```json
{
  "error": "Invalid JSON payload"
}
```

`404 Not Found`:

```json
{
  "error": "Dataset not found"
}
```

---

#### Remove Dataset

Unregisters the dataset from the server (does not delete files from disk).

`DELETE /v1/datasets/<id>`

**Response: 200 OK**

```json
{
  "status": "deleted"
}
```

**Error Response:**

`404 Not Found`:

```json
{
  "error": "Dataset not found"
}
```

---

#### List Files

Returns a list of all files inside the dataset directory.

`GET /v1/datasets/<id>/files`

**Response: 200 OK**

```json
{
  "files": [
    {
      "base": "image001",
      "name": "image001.png",
      "path": "",
      "files": [
        {
          "path": "image001.png",
          "name": "image001.png",
          "size_bytes": 1048576,
          "extension": ".png",
          "file_type": "image"
        }
      ],
      "caption": "",
      "caption_path": ""
    },
    {
      "base": "image002",
      "name": "image002.jpg",
      "path": "subfolder",
      "files": [
        {
          "path": "subfolder/image002.jpg",
          "name": "image002.jpg",
          "size_bytes": 524288,
          "extension": ".jpg",
          "file_type": "image"
        }
      ],
      "caption": "",
      "caption_path": ""
    }
  ]
}
```

**Error Responses:**

`404 Not Found` - Dataset not found:

```json
{
  "error": "Dataset not found"
}
```

`404 Not Found` - Path no longer exists:

```json
{
  "error": "Dataset path no longer exists on disk"
}
```

---

#### Download/Serve File

Serves the raw binary file with appropriate Content-Type header. Protected against directory traversal attacks.

`GET /v1/datasets/<id>/files/<relative/file/path.ext>`

**Response: 200 OK**

Returns the raw binary file content with the appropriate `Content-Type` header (e.g., `image/png`, `image/jpeg`, `audio/wav`, etc.).

**Error Responses:**

`403 Forbidden` - Path traversal attempt:

```json
{
  "error": "Access denied: Path traversal detected"
}
```

`404 Not Found` - Dataset not found:

```json
{
  "error": "Dataset not found"
}
```

`404 Not Found` - File not found:

```json
{
  "error": "File not found"
}
```

---

### Jobs (Command Execution)

You can trigger _any_ datasety command (e.g., `resize`, `caption`, `train`) remotely. The parameters in the `args` dictionary map directly to the command-line flags.

#### Start a Job

`POST /v1/jobs`

**Request:**

```json
{
  "command": "resize",
  "args": {
    "input": "/path/to/raw",
    "output": "/path/to/resized",
    "resolution": "512x512",
    "crop-position": "top"
  }
}
```

**Response: 202 Accepted**

```json
{
  "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "started"
}
```

**Error Responses:**

`400 Bad Request` - Missing command:

```json
{
  "error": "Missing 'command' in payload"
}
```

`400 Bad Request` - Invalid JSON:

```json
{
  "error": "Invalid JSON payload"
}
```

`500 Internal Server Error` - Failed to start:

```json
{
  "error": "Failed to start job: [error details]"
}
```

---

#### List Jobs

`GET /v1/jobs`

**Response: 200 OK**

```json
{
  "jobs": [
    {
      "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
      "command": "resize",
      "argv": [
        "/usr/bin/python3",
        "-m",
        "datasety",
        "resize",
        "--input",
        "/path/to/raw",
        "--output",
        "/path/to/resized",
        "--resolution",
        "512x512",
        "--crop-position",
        "top"
      ],
      "status": "running",
      "output": ["Processing file 1...", "Processing file 2..."],
      "exit_code": null,
      "started_at": 1709633000.123456,
      "ended_at": null
    },
    {
      "id": "b2c3d4e5-f6a7-8901-bcde-f23456789012",
      "command": "caption",
      "argv": [
        "/usr/bin/python3",
        "-m",
        "datasety",
        "caption",
        "--input",
        "/path/to/images"
      ],
      "status": "done",
      "output": ["Generated 100 captions"],
      "exit_code": 0,
      "started_at": 1709632800.123456,
      "ended_at": 1709632900.789012
    }
  ]
}
```

---

#### Check Job Status & Logs

`GET /v1/jobs/<id>`

**Response: 200 OK** (Running job):

```json
{
  "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "command": "resize",
  "argv": [
    "/usr/bin/python3",
    "-m",
    "datasety",
    "resize",
    "--input",
    "/path/to/raw",
    "--output",
    "/path/to/resized"
  ],
  "status": "running",
  "output": ["Processing images...", "Resized image001.png"],
  "exit_code": null,
  "started_at": 1709633000.123456,
  "ended_at": null
}
```

**Response: 200 OK** (Completed job):

```json
{
  "id": "b2c3d4e5-f6a7-8901-bcde-f23456789012",
  "command": "caption",
  "argv": [
    "/usr/bin/python3",
    "-m",
    "datasety",
    "caption",
    "--input",
    "/path/to/images"
  ],
  "status": "done",
  "output": ["Processing images...", "Generated 100 captions"],
  "exit_code": 0,
  "started_at": 1709632800.123456,
  "ended_at": 1709632900.789012
}
```

**Error Response:**

`404 Not Found`:

```json
{
  "error": "Job not found"
}
```

---

#### Cancel a Job

Terminates a running job.

`DELETE /v1/jobs/<id>`

**Response: 200 OK**

```json
{
  "status": "cancelled"
}
```

**Error Responses:**

`404 Not Found`:

```json
{
  "error": "Job not found"
}
```

`400 Bad Request` - Job not running:

```json
{
  "error": "Job is not running"
}
```

---

### Commands Schema

Get all available command schemas with their parameters. Useful for building dynamic UIs.

`GET /v1/commands`

**Response: 200 OK**

```json
{
  "commands": {
    "resize": {
      "description": "Resize and crop images to target resolution",
      "params": [
        { "name": "input", "help": "Input directory containing images", "required": false },
        { "name": "output", "help": "Output directory for processed images", "required": false },
        { "name": "resolution", "help": "Target resolution as WIDTHxHEIGHT", "required": false },
        { "name": "crop_position", "help": "Position to keep when cropping", "required": false, "choices": ["top", "center", "bottom", "left", "right"], "default": "center" }
      ]
    },
    "caption": { ... },
    ...
  }
}
```

---

### Caption Management

#### Get Caption

Get the caption/instruction text for an image (reads from companion .txt file).

`GET /v1/datasets/<id>/caption/<relative/path/image.ext>`

**Response: 200 OK**

```json
{
  "caption": "add a winter hat to the person",
  "path": "edit_001.txt"
}
```

#### Save Caption

Save the caption/instruction text for an image (writes to companion .txt file).

`PUT /v1/datasets/<id>/caption/<relative/path/image.ext>`

**Request:**

```json
{
  "caption": "change background to blue"
}
```

**Response: 200 OK**

```json
{
  "status": "saved",
  "path": "edit_001.txt"
}
```

---

#### Save Metadata

Update the text in `metadata.csv` for an audio file (Piper/LJSpeech format).

`PUT /v1/datasets/<id>/metadata/<relative/path/audio.wav>`

**Request:**

```json
{
  "text": "Updated transcription text"
}
```

**Response: 200 OK**

```json
{
  "status": "saved",
  "path": "metadata.csv"
}
```

---

### Dataset File Queries

#### List Files with Folder Filter

Filter files by subfolder and optionally group by base filename.

`GET /v1/datasets/<id>/files?folder=<subfolder>&group=true`

**Query Parameters:**

- `folder` (optional) - Filter to specific subfolder
- `group` (optional) - When `true`, groups files by base name (for *input/*target/\*mask naming conventions)

**Response: 200 OK** (without group)

```json
{
  "files": [
    {
      "base": "edit_001",
      "name": "edit_001_input.png",
      "path": "",
      "files": [
        {
          "path": "edit_001_input.png",
          "name": "edit_001_input.png",
          "size_bytes": 234837,
          "extension": ".png",
          "file_type": "input"
        },
        {
          "path": "edit_001_target.png",
          "name": "edit_001_target.png",
          "size_bytes": 117743,
          "extension": ".png",
          "file_type": "target"
        },
        {
          "path": "edit_001.txt",
          "name": "edit_001.txt",
          "size_bytes": 17,
          "extension": ".txt"
        }
      ],
      "caption": "add a winter hat",
      "caption_path": "edit_001.txt"
    }
  ]
}
```

**Response: 200 OK** (with group=true)

```json
{
  "pairs": [
    {
      "base": "edit_001",
      "input": "edit_001_input.png",
      "target": "edit_001_target.png",
      "mask": null,
      "image": null,
      "caption": "add a winter hat",
      "all_files": [...]
    }
  ]
}
```

**Supported naming conventions:**

- `*_input` or `*_start` → original image
- `*_target` or `*_end` → edited result
- `*_mask` → binary mask (optional)
- `*.txt` → caption/instruction file

---

### Generic Errors

`404 Not Found` - Unknown endpoint:

```json
{
  "error": "Endpoint not found"
}
```
