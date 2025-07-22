import { Stack, Typography, Divider, Box } from "@mui/material";

const Header = () => {
  return (
    <Stack direction="column" spacing={3} width="100%" alignItems="center">
      <Typography variant="h4">
        Patent Similarity Search for Quantum Photonic Circuits Utilizing{" "}
        <Box
          component="span"
          sx={{
            background: "linear-gradient(to right, #80cbc4, #00695c)", // light to dark teal
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
            fontWeight: "bold",
          }}
        >
          Vector Similarity
        </Box>
      </Typography>
      <Typography variant="h5" color="#505050ff">
        Developed By Eric Jacobson, Wilber Huang, and Willian Han
      </Typography>
      <Divider sx={{ width: "70%", paddingTop: 4 }} />
    </Stack>
  );
};
export default Header;
