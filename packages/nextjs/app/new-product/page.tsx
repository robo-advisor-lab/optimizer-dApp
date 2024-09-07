"use client";

import React from "react";
import { useState } from "react";
import Box from "@mui/material/Box";
import Checkbox from "@mui/material/Checkbox";
import FormControl from "@mui/material/FormControl";
import FormControlLabel from "@mui/material/FormControlLabel";
import FormGroup from "@mui/material/FormGroup";
import InputLabel from "@mui/material/InputLabel";
import MenuItem from "@mui/material/MenuItem";
import Select, { SelectChangeEvent } from "@mui/material/Select";
import TextField from "@mui/material/TextField";

const Page = () => {
  const [durationForm, setDurationForm] = useState("");
  const [frequenceForm, setFrequenceForm] = useState("");
  const [numAdress, setNumAdres] = useState(1);
  const elements = [];

  const handleClickAdressInputPlus = () => {
    setNumAdres(numAdress + 1);
  };
  const handleClickAdressInputLesss = () => {
    if (numAdress > 1) setNumAdres(numAdress - 1);
  };

  const handleChangeDuration = (event: SelectChangeEvent) => {
    setDurationForm(event.target.value as string);
  };

  const handleChangeFrequence = (event: SelectChangeEvent) => {
    setFrequenceForm(event.target.value as string);
  };

  for (let index = 0; index < numAdress; index++) {
    elements.push(
      <Box component="form" sx={{ "& > :not(style)": { m: 1, width: "25ch" } }} noValidate autoComplete="off">
        <TextField
          id="outlined-basic"
          label={"Cripto Address " + (index + 1)}
          variant="outlined"
          name={"adress" + index}
        />
      </Box>,
    );
  }
  return (
    <div>
      <div className="container mx-auto px-4 py-16">
        <div className="flex flex-col md:flex-row items-center justify-between">
          <div className="md:w-1/2 mb-8 md:mb-0">
            <h2 className="text-4xl font-bold mb-4">Unlock Your Financial Potential</h2>
            <p className="text-gray-700 leading-relaxed">
              Our investment product is designed to help you achieve your financial goals. With a diversified portfolio
              and expert management, we&apos;ll guide you towards long-term success.
            </p>
            <button className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded mt-6">
              Learn More
            </button>
          </div>
          <div className="md:w-1/2">
            <div className="bg-blue-100 px-6 py-4 mt-3 rounded-lg">
              <h1>DAO Governance</h1>
              <p>¿Cón que cripto prefeires trabajar?</p>
              <FormGroup>
                <FormControlLabel control={<Checkbox defaultChecked />} label="Label" />
                <FormControlLabel control={<Checkbox defaultChecked />} label="Label" />
                <FormControlLabel control={<Checkbox defaultChecked />} label="Label" />
                <FormControlLabel control={<Checkbox defaultChecked />} label="Label" />
              </FormGroup>
              <div className="flex gap-3">
                <p>Dirección Cripto</p>
                <div
                  className="bg-blue-300 text-blue-800 px-3 rounded-full flex items-center justify-center cursor-pointer select-none"
                  onClick={handleClickAdressInputPlus}
                >
                  +
                </div>
                <div
                  className="bg-red-300 text-red-800 px-3 rounded-full flex items-center justify-center cursor-pointer select-none"
                  onClick={handleClickAdressInputLesss}
                >
                  -
                </div>
              </div>
              <div>{elements}</div>
              <p>Duración</p>
              <Box sx={{ minWidth: 120 }}>
                <FormControl fullWidth>
                  <InputLabel id="demo-simple-select-label">Duracion</InputLabel>
                  <Select
                    labelId="demo-simple-select-label"
                    id="demo-simple-select"
                    value={durationForm}
                    label="Duracion"
                    onChange={handleChangeDuration}
                  >
                    <MenuItem value={10}>Ten</MenuItem>
                    <MenuItem value={20}>Twenty</MenuItem>
                    <MenuItem value={30}>Thirty</MenuItem>
                  </Select>
                </FormControl>
              </Box>
              <p>Frecuencia</p>
              <Box sx={{ minWidth: 120 }}>
                <FormControl fullWidth>
                  <InputLabel id="demo-simple-select-label">Frecuencia</InputLabel>
                  <Select
                    labelId="demo-simple-select-label"
                    id="demo-simple-select"
                    value={frequenceForm}
                    label="Frecuencia"
                    onChange={handleChangeFrequence}
                  >
                    <MenuItem value={10}>Ten</MenuItem>
                    <MenuItem value={20}>Twenty</MenuItem>
                    <MenuItem value={30}>Thirty</MenuItem>
                  </Select>
                </FormControl>
              </Box>
            </div>
          </div>
        </div>

        {/* Additional sections can be added here for features, benefits, testimonials, etc. */}
        <div className="mt-16">{/* ... more content ... */}</div>
      </div>
    </div>
  );
};

export default Page;
