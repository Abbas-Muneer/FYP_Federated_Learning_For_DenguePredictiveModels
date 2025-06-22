import axios from "axios";

const instance = axios.create({
    baseURL: "http://localhost:5000", // Flask backend URL
    withCredentials: true    
});

export default instance;
