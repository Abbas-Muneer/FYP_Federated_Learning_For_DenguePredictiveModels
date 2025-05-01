import React, { useState } from "react";
import axios from "../services/api";

const UpdateConfig = () => {
    const [config, setConfig] = useState({
        epochs: "",
        rounds: "",
        learning_rate: "",
        momentum: ""
    });
    const [message, setMessage] = useState("");

    const handleChange = (e) => {
        setConfig({ ...config, [e.target.name]: e.target.value });
    };

    const handleUpdate = async () => {
        console.log(config); 
        try {
            const response = await axios.post("/admin/update-config", config);
            setMessage(response.data.message);
        } catch (error) {
            setMessage("Failed to update configuration.");
        }
    };

    return (
        <div className="flex flex-col p-2 ">
            <h2 className="poppins-semibold text-md text-center">Update Configurations</h2>
            <form className="text-xs poppins-medium flex flex-col gap-1 p-2">
                <label>
                    Epochs:
                    <input type="number" name="epochs" value={config.epochs} onChange={handleChange} />
                </label>
                <br />
                <label>
                    Rounds:
                    <input type="number" name="rounds" value={config.rounds} onChange={handleChange} />
                </label>
                <br />
                <label>
                    Learning Rate:
                    <input type="number" step="0.01" name="learning_rate" value={config.learning_rate} onChange={handleChange} />
                </label>
                <br />
                <label>
                    Momentum:
                    <input type="number" step="0.01" name="momentum" value={config.momentum} onChange={handleChange} />
                </label>
                <br />

                <div className="flex items-center justify-center">
                    <button className="p-2 bg-[#1b1b1b] rounded-md text-white poppins-medium text-sm w-48" type="button" onClick={handleUpdate}>
                        Update Config
                    </button>
                </div>
                
            </form>
            {message && <p>{message}</p>}
        </div>
    );
};

export default UpdateConfig;
