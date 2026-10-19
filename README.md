# Regression Explorer — Dark UI

## Run in VS Code
1. Extract this ZIP and open the `regression_explorer_dark` folder in VS Code.
2. Open Terminal → New Terminal.
3. Install dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```
4. Run:
   ```bash
   python -m streamlit run app.py
   ```
5. Open the local address shown in the terminal (usually http://localhost:8501).

## Included
- Dark UI inspired by the supplied reference screenshot
- Sidebar dataset, model, input feature, and test split controls
- Overview, Model Comparison, Regression Plot, Residuals, and Conclusions tabs
- MAE, MSE, RMSE, R², comparison charts, residual charts, and CSV export

The first California Housing load requires internet access. The UI is built with Streamlit.
