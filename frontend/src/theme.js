// src/theme/theme.js
import { createTheme } from '@mui/material/styles';
import "@fontsource/inter"

const theme = createTheme({
  palette: {
    mode: 'dark',
    background: {
      default: '#121212',   // very dark background
      paper: '#1E1E1E',     // slightly lighter for cards/panels
    },
    primary: {
      main: '#076f55ff',      // subtle teal/green accent
    },
    secondary: {
      main: '#313131ff',      // muted gray for secondary elements
    },
    text: {
      primary: '#e0e0e0',   // soft white text
      secondary: '#aaaaaa', // softer secondary text
    },
  },
  typography: {
    fontFamily: '"Inter", sans-serif',
    fontWeightRegular: 400,
    fontWeightMedium: 600,
    fontWeightBold: 700,
  },
});

export default theme;
