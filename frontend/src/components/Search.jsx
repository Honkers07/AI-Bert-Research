import {
  TextField,
  Button,
  Stack,
  Card,
  CardContent,
  Typography,
  ButtonGroup,
  Slider,
} from "@mui/material";
import { useState } from "react";
import axios from "axios";

const Search = () => {
  // Patent inputs
  const [title, setTitle] = useState("");
  const [claim, setClaim] = useState("");
  const [description, setDescription] = useState("");

  // Settings
  const [showSettings, setShowSettings] = useState(false);
  const [type, setType] = useState("cosine");
  const [claimWeight, setClaimWeight] = useState(7);
  const [numberOfTops, setNumberOfTops] = useState(5);

  // Backend
  const [results, setResults] = useState([]);
  const [error, setError] = useState(null);

  const handleSubmit = async () => {
    setError(null);
    try {
      const response = await axios.post("http://localhost:8000/search", {
        query_title: title,
        query_claim: claim,
        query_description: description,
        type: type,
        claimWeight: claimWeight / 10, // convert to decimal
        topK: numberOfTops,
      });
      setResults(response.data.results);
    } catch (err) {
      setError("Search failed. Please try again.");
      console.error("Search failed:", err);
    }
  };

  return (
    <Stack direction="column" spacing={4}>
      <TextField
        label="Title"
        variant="outlined"
        value={title}
        onChange={(e) => setTitle(e.target.value)}
      />
      <TextField
        label="Claim"
        variant="outlined"
        value={claim}
        onChange={(e) => setClaim(e.target.value)}
      />
      <TextField
        label="Description"
        variant="outlined"
        value={description}
        onChange={(e) => setDescription(e.target.value)}
      />
      <Button variant="contained" onClick={handleSubmit} sx={{ height: 47 }}>
        Search
      </Button>
      <Button
        variant={showSettings ? "contained" : "outlined"}
        onClick={() => setShowSettings(!showSettings)}
        sx={{
          height: 47,
          backgroundColor: showSettings ? "#2c2c2c" : "transparent",
          color: "#ffffff",
          borderColor: showSettings ? "transparent" : "#2c2c2c",
          "&:hover": {
            backgroundColor: showSettings ? "#212121" : "transparent",
            borderColor: showSettings ? "transparent" : "#454545ff",
          },
        }}
      >
        Settings
      </Button>

      {showSettings && (
        <Stack direction="column" spacing={4} sx={{ color: "#fff" }}>
          <ButtonGroup>
            <Button
              onClick={() => setType("cosine")}
              variant={type === "cosine" ? "contained" : "outlined"}
              fullWidth
            >
              Cosine
            </Button>
            <Button
              onClick={() => setType("euclidean")}
              variant={type === "euclidean" ? "contained" : "outlined"}
              fullWidth
            >
              Euclidean
            </Button>
          </ButtonGroup>

          <Typography sx={{ color: "#fff" }}>
            Claim Weight: {claimWeight === 0 ? "" : claimWeight}0% | Description
            Weight: {claimWeight === 10 ? "" : 10 - claimWeight}0%
          </Typography>

          <Slider
            value={claimWeight}
            onChange={(e) => setClaimWeight(e.target.value)}
            min={0}
            max={10}
            sx={{
              color: "#2c2c2c",
            }}
          />

          <Typography sx={{ color: "#fff" }}>
            Number of Similar Patents Returned: {numberOfTops}
          </Typography>

          <Slider
            value={numberOfTops}
            onChange={(e) => setNumberOfTops(e.target.value)}
            min={1}
            max={20}
            sx={{
              color: "#2c2c2c",
            }}
          />
        </Stack>
      )}

      {error && <Typography color="error">{error}</Typography>}

      {results.length > 0 &&
        results.map(({ rank, publication_number, title, score, date }) => (
          <Card key={publication_number} variant="outlined" sx={{ mt: 2 }}>
            <CardContent>
              <Typography variant="h6">
                {rank}. {title}
              </Typography>
              <Typography variant="body2">
                Publication #: {publication_number}
              </Typography>
              <Typography variant="body2">Date: {date}</Typography>
              <Typography variant="body2">Score: {score.toFixed(4)}</Typography>
            </CardContent>
          </Card>
        ))}
    </Stack>
  );
};

export default Search;
