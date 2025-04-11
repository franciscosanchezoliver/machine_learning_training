course: https://campus.datacamp.com/courses/introduction-to-llms-in-python/getting-started-with-large-language-models-llms?ex=1


start api point:
uvicorn summarizer_api_point:app --host 0.0.0.0 --port 9999


Example of calling the api:
```powershell
Invoke-RestMethod -Method POST `
  -Uri "http://<YOUR_PI_IP>:9999/summarize" `
  -ContentType "application/json" `
  -Body (@{
      text = "Walking amid Gion's Machiya houses is a mesmerizing experience. The beautifully preserved structures exude an old-world charm that transports visitors back in time. The glow of lanterns lining the narrow streets adds to the enchanting ambiance."
      max_length = 50
  } | ConvertTo-Json)

```


launch.json for debugging in vscode.
```json
{
    "version": "0.2.0",
    "configurations": [
      {
        "name": "FastAPI: Run with Uvicorn",
        "type": "python",
        "request": "launch",
        "module": "uvicorn",
        "args": [
          "summarizer_api_point:app",
          "--host", "0.0.0.0",
          "--port", "9999",
          "--reload"
        ],
        "jinja": true,
        "justMyCode": true
      }
    ]
  }


```