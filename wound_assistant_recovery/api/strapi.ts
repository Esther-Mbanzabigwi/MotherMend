import axios from "axios";

const strapi = axios.create({
  baseURL: `https://wound-assitant-backend.onrender.com/api`,
});

export default strapi;
