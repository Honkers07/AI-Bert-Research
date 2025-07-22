import { useState } from "react";
import { CssBaseline, Stack, Box, ThemeProvider } from "@mui/material";
import theme from "./theme";
import Search from "./components/search";
import Header from "./components/Header";

function App() {
  return (
    <Box
      sx={{
        display: "flex",
        flexDirection: "column",
        padding: 4,
        width: "100%",
        mt: 5
      }}
    >
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <Stack direction="column" spacing={7}>
          <Header />
          <Search />
        </Stack>
      </ThemeProvider>
    </Box>
  );
}

export default App;
