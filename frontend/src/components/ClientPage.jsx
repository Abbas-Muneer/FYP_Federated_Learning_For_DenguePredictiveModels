import React, { useEffect, useState } from "react";
import axios from "../services/api";
import { useNavigate } from "react-router-dom";

const ClientPage = ({ user }) => {  //  Ensure user is passed as a prop
    const [file, setFile] = useState(null);
    const [message, setMessage] = useState("");
    const navigate = useNavigate();
    const [summary, setSummary] = useState({ accuracy: null, loss: null, contribution_score: null });
    const [globalSummary, setGlobalSummary] = useState({
        global_accuracy: null,
        global_loss: null,
        global_last_trained: null
    });

    //  Ensure user exists before rendering
    useEffect(() => {
        if (!user) {
            console.warn("No user found, redirecting to login...");
            navigate("/"); // Redirect to login if user is not logged in
            return;
        }
        fetchClientSummary();
        fetchGlobalSummary();
    }, [user, navigate]);

    //  Fetch Global Summary from API
    const fetchGlobalSummary = async () => {
        try {
            const response = await axios.get("/admin/summary");
            setGlobalSummary({
                global_accuracy: response.data.accuracy,
                global_loss: response.data.loss,
                global_last_trained: response.data.last_trained
            });
        } catch (error) {
            console.error("Failed to load global summary:", error);
            setMessage("Failed to load global summary.");
        }
    };

    //  Fetch Client Summary from API
    const fetchClientSummary = async () => {
        try {
            const response = await axios.get("/client/summary");
            setSummary({
                accuracy: response.data.accuracy,
                loss: response.data.loss,
                contribution_score: response.data.contribution_score
            });
        } catch (error) {
            console.error("Failed to fetch client summary:", error);
            setMessage("Failed to fetch client summary.");
        }
    };

    //  Handle File Selection
    const handleFileChange = (e) => {
        setFile(e.target.files[0]);
    };

    //  Upload Dataset
    const handleUpload = async () => {
        if (!file) {
            setMessage("Please select a file before uploading.");
            return;
        }

        const formData = new FormData();
        formData.append("client_id", user?.name || "Unknown_Client");
        formData.append("file", file);

        try {
            const response = await axios.post("/client/add-dataset", formData);
            setMessage(`Dataset uploaded successfully.`);
        } catch (error) {
            console.error("Error uploading dataset:", error);
            setMessage(error.response?.data?.error || "An error occurred while uploading the dataset.");
        }
    };

    //  Download Global Model
    const handleDownloadModel = async () => {
        try {
            const response = await axios.get("/client/download-model", { responseType: "blob" });
            const url = window.URL.createObjectURL(new Blob([response.data]));
            const link = document.createElement("a");
            link.href = url;
            link.setAttribute("download", "global_model.pth");
            document.body.appendChild(link);
            link.click();
            link.remove();
        } catch (error) {
            console.error("Failed to download global model:", error);
            setMessage("Failed to download the global model.");
        }
    };

    return (
        <div className="h-screen text-[#1b1b1b] flex flex-col items-center justify-center">
            <div>
                <h1>Welcome, {user?.name || "Client"}</h1>
            </div>
            <div className="flex gap-5 mt-5">
                <input className="border-gray-200 border rounded-md text-xs p-2 underline cursor-pointer" type="file" onChange={handleFileChange} />
                <button className="bg-[#1b1b1b] text-white rounded-md p-2 text-sm" onClick={handleUpload}>Add Dataset</button>
                <button className="bg-[#1b1b1b] text-white rounded-md p-2 text-sm" onClick={handleDownloadModel}>Download Global Model</button>
            </div>

            <p className="text-sm underline text-[#1b1b1b] mt-5">{message}</p>

            <div className="flex gap-5 mt-5 w-full p-20">
                <div className="flex flex-col w-1/2 gap-5 border border-gray-200 rounded-md shadow-md p-2">
                    <h2 className="text-md text-center text-[#1b1b1b]">Client Summary</h2>
                    <p className="text-xs">Accuracy: {summary.accuracy ?? "N/A"}%</p>
                    <p className="text-xs">Loss: {summary.loss ?? "N/A"}%</p>
                    <p className="text-xs">Contribution Score: {summary.contribution_score ?? "N/A"}</p>
                </div>

                <div className="flex flex-col border border-gray-200 shadow-md gap-5 rounded-md p-2 w-1/2">
                    <h2 className="text-md text-center">Global Summary</h2>
                    <p className="text-xs">Accuracy: {globalSummary.global_accuracy ?? "N/A"}%</p>
                    <p className="text-xs">Loss: {globalSummary.global_loss ?? "N/A"}</p>
                    <p className="text-xs">Last Trained: {globalSummary.global_last_trained ?? "N/A"}</p>
                </div>
            </div>
        </div>
    );
};

export default ClientPage;
