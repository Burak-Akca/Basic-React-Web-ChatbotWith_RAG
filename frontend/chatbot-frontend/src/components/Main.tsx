import { useState } from "react";
import { TextField, Button, Typography, Container } from "@mui/material";
import axios, { AxiosError } from "axios";

const ApiFetcher = () => {
  const [inputValue, setInputValue] = useState<string>("");
  const [responseData, setResponseData] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const handleFetch = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await axios.post<{ answer: string }>("http://localhost:8000/rag", {
        question: inputValue,
      });
      setResponseData(response.data.answer);
    } catch (err) {
      const axiosError = err as AxiosError;  // TypeScript hatası için axios error tipi
      console.error(axiosError.response ? axiosError.response.data : axiosError.message);

      setError("Veri alınırken hata oluştu");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container maxWidth="sm" style={{ textAlign: "center", marginTop: "20px" }}>
      <TextField
        label="Write something"
        variant="outlined"
        fullWidth
        value={inputValue}
        onChange={(e) => setInputValue(e.target.value)}
        style={{ marginBottom: "10px" }}
      />
      <Button variant="contained" color="primary" onClick={handleFetch} disabled={loading}>
        {loading ? "Loading..." : "Submit"}
      </Button>
      {responseData && (
        <Typography variant="h6" style={{ marginTop: "15px" }}>
          {responseData}
        </Typography>
      )}
      {error && (
        <Typography color="error" variant="body1" style={{ marginTop: "15px" }}>
          {error}
        </Typography>
      )}
    </Container>
  );
};

export default ApiFetcher;
