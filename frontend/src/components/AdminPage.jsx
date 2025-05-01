import React, { useEffect, useState } from "react";
import axios from "../services/api";
import UpdateConfig from "./UpdateConfig";

const AdminPage = () => {
    const [clients, setClients] = useState([]);
    const [error, setError] = useState("");
    const [isTraining, setIsTraining] = useState(false);  
    const [summary, setSummary] = useState({
        accuracy: null,
        loss: null,
        last_trained: null
    });

    // Fetch global summary
    const fetchSummary = async () => {
        try {
            const response = await axios.get("/admin/summary");
            setSummary({
                accuracy: response.data.accuracy,
                loss: response.data.loss,
                last_trained: response.data.last_trained
            });
        } catch (error) {
            setError("Failed to load global summary.");
        }
    };

    // Fetch client details
    const fetchClients = async () => {
        try {
            const response = await axios.get("/admin/get-clients");
            setClients(response.data.clients);
        } catch (error) {
            setError("Failed to load client data.");
        }
    };

    useEffect(() => {
        fetchClients();
        fetchSummary();
    }, []);

    const handleStartTraining = async () => {
        setIsTraining(true); 
        try {
            const response = await axios.post("/admin/start-fl-training");
            alert(`Training completed. Accuracy: ${response.data.accuracy}, Loss: ${response.data.loss}`);
            await fetchSummary();
            await fetchClients();
        } catch (error) {
            alert("Failed to start training.");
        } finally {
            setIsTraining(false); 
        }
    };

    return (
        <div className="h-screen text-[#1b1b1b] flex flex-col items-center justify-center">
            <h1 className="text-4xl poppins-semibold">Admin Page</h1>
            <button 
                className="mt-2 bg-[#1b1b1b] text-white rounded-md p-2 poppins-regular text-sm flex items-center justify-center"
                onClick={handleStartTraining}
                disabled={isTraining}  
            >
                {isTraining ? "Training..." : "Start Training"} {/* <-- Dynamic button text */}
            </button>

            {error && <p className="mt-2 text-[#1b1b1b] poppins-medium text-xs" style={{ color: "red" }}>{error}</p>}

            <div className="flex gap-5 mt-5 w-full p-10">
                <div className="flex flex-col border border-gray-200 shadow-md gap-5 rounded-md p-2 w-1/3">
                    <h2 className="poppins-semibold text-md text-center">Global Summary</h2>
                    <p className="poppins-medium text-xs">Accuracy: {summary.accuracy ?? "N/A"}%</p>
                    <p className="poppins-medium text-xs">Loss: {summary.loss ?? "N/A"}</p>
                    <p className="poppins-medium text-xs">Last Trained: {summary.last_trained ?? "N/A"}</p>
                </div>
                <div className="flex flex-col border border-gray-200 w-1/3 shadow-md rounded-md p-2">
                    <h2 className="poppins-semibold text-md text-center mb-2">Client Details</h2>
                    <table className="border border-gray-200 rounded-md ">
                        <thead>
                            <tr className="flex gap-2 p-2 poppins-regular text-xs">
                                <th>Client ID</th>
                                <th>Dataset Name</th>
                                <th>Accuracy</th>
                                <th>Loss</th>
                                <th>Contribution Score</th>
                            </tr>
                        </thead>
                        <tbody className="border border-gray-200 rounded-md">
                            {clients.map((client) => (
                                <tr className="flex gap-2 p-2 poppins-regular text-xs" key={client.id}>
                                    <td>{client.client_id}</td>
                                    <td>{client.dataset_name}</td>
                                    <td>{client.accuracy}</td>
                                    <td>{client.loss}</td>
                                    <td>{client.contribution_score}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
                <div className="border border-gray-200 shadow-md w-1/3 rounded-md ">
                    <UpdateConfig />
                </div>
            </div>
        </div>
    );
};

export default AdminPage;